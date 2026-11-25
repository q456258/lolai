"""Turns game state into advice: best champion in champ select, best augment on screen."""

from __future__ import annotations

from collections.abc import Iterator

from ollama import chat
from pydantic import BaseModel

from lolai.domain.enums.model import Model
from lolai.integrations.riot.models import ChampSelect, LiveGame
from lolai.integrations.riot.champions import champ_slug
from lolai.integrations.riot import mayhemmeta
from lolai.integrations.riot.opgg.opgg import connect, match_name, rank_cards


# ---- champion selection ----------------------------------------------------------------

class ChampPick(BaseModel):
    champion: str
    reason: str


class ChampAdvice(BaseModel):
    best: ChampPick
    alternatives: list[ChampPick]
    warnings: list[str] = []


_CHAMP_SYSTEM = (
    "You are a League of Legends ARAM Mayhem coach. Recommend the single best champion for the "
    "player to end up on, judging team composition (damage split AP/AD, frontline, engage, "
    "crowd control, scaling) and counters to the known enemies. You may ONLY recommend champions "
    "from the provided candidate list. Be concise: one sentence per reason."
)


def _champ_prompt(cs: ChampSelect) -> str:
    if cs.pickable:
        mode = f"Draft pick, role: {cs.my_position or 'unknown'}. Candidates (pickable): "
        candidates = cs.pickable
    elif cs.bench:
        mode = f"ARAM swap. Rerolls left: {cs.rerolls_left}. Candidates (current pick + bench): "
        candidates = [c for c in [cs.my_champion, *cs.bench] if c]
    else:
        mode = "Candidates: "
        candidates = [cs.my_champion] if cs.my_champion else []
    return (
        f"{mode}{', '.join(candidates) or 'none'}\n"
        f"Currently selected/hovered: {cs.my_champion or 'nothing'}\n"
        f"Allies: {', '.join(cs.allies) or 'unknown'}\n"
        f"Enemies: {', '.join(cs.enemies) or 'hidden/unknown'}\n"
        f"Banned: {', '.join(cs.bans) or 'none'}"
    )


def advise_champ_select(cs: ChampSelect) -> ChampAdvice:
    res = chat(
        model=Model.QWEN.value,
        messages=[
            {"role": "system", "content": _CHAMP_SYSTEM},
            {"role": "user", "content": _champ_prompt(cs)},
        ],
        format=ChampAdvice.model_json_schema(),
        think=False,
    )
    return ChampAdvice.model_validate_json(res.message.content)


def format_champ_select(a: ChampAdvice) -> str:
    lines = [f"★ BEST: {a.best.champion}", f"  {a.best.reason}", ""]
    for alt in a.alternatives:
        lines.append(f"• {alt.champion}: {alt.reason}")
    lines += [f"⚠ {w}" for w in a.warnings]
    return "\n".join(lines)


# ---- augments -----------------------------------------------------------------------------
# Three stages so the player sees something quickly:
#   1. read_augment_names  - small vision call, names only            (a few seconds)
#   2. rank_augments       - database lookup + fuzzy verification     (instant)
#   3. stream_augment_advice - text-only model, streamed token by token (slow, shown progressively)

# Your GPU cannot hold the vision model and the text model at once, so using two models makes every
# F8 press evict one and reload the other (measured: 6.5s). The vision model also writes the advice.
AUGMENT_MODEL = Model.QWEN_VL.value
KEEP_ALIVE = "30m"  # Ollama's default is 5 minutes, which would unload the model mid-game


def warm_up(model: str = AUGMENT_MODEL) -> None:
    """Load the model into memory ahead of time (e.g. during the loading screen)."""
    chat(model=model, messages=[], keep_alive=KEEP_ALIVE)


class CardNames(BaseModel):
    names: list[str]


_READ_SYSTEM = (
    "The image is a League of Legends screenshot. If augment cards to choose from are visible, list the "
    "name printed on each card exactly as written, left to right. Return only names, nothing else. "
    "If no augment cards are visible, return an empty list."
)


def read_augment_names(screenshot: bytes) -> list[str]:
    res = chat(
        model=AUGMENT_MODEL,
        keep_alive=KEEP_ALIVE,
        messages=[
            {"role": "system", "content": _READ_SYSTEM},
            {"role": "user", "content": "List the augment card names.", "images": [screenshot]},
        ],
        format=CardNames.model_json_schema(),
    )
    return CardNames.model_validate_json(res.message.content).names


def rank_augments(names: list[str], game: LiveGame | None, owned: list[str] | None = None) -> list[dict]:
    """Keep only names that match a known augment (exact, or fuzzy for misreads), then rank them
    for the player's champion from the database. Unknown text is not an augment and is dropped.
    Each card also gets card["mm"]: mayhemmeta win rates (see mayhemmeta.card_stats)."""
    con = connect()
    try:
        if not con.execute("SELECT 1 FROM augments LIMIT 1").fetchone():
            raise RuntimeError("Augment database is empty. Run: python -m lolai.integrations.riot.opgg.opgg augments")
        verified = list(dict.fromkeys(n for n in (match_name(con, x) for x in names) if n))
        slug = champ_slug(game.my_champion) if game else None
        cards = rank_cards(con, slug, verified)
        mayhemmeta.connect().close()  # creates the mm_ tables if the scraper never ran
        stats = mayhemmeta.card_stats(con, [c["name"] for c in cards],
                                      game.my_champion if game else None, owned)
        for c in cards:
            c["mm"] = stats.get(c["name"], {})
        return cards
    finally:
        con.close()


TIER_LETTERS = "SABCDE"  # op.gg tier 0..5


def tier_letter(tier: int | None) -> str:
    if tier is None:
        return "?"
    return TIER_LETTERS[tier] if 0 <= tier < len(TIER_LETTERS) else str(tier)


def _pct(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:+.1f}" if signed else f"{v:.1f}%"


def card_stat_lines(c: dict, who: str) -> list[str]:
    """The six stats for one card, as display lines (shared by the overlay and the AI prompt)."""
    mm = c.get("mm") or {}
    if mm.get("win_rate") is None:
        return ["   no mayhemmeta data (run: python -m lolai.integrations.riot.mayhemmeta load)"]
    lines = [f"   Tier {mm.get('tier') or '?'}  ·  Win rate {_pct(mm['win_rate'])}  ·  "
             f"Pick rate {_pct(mm.get('pick_rate'))}"]
    if mm.get("champ_win_rate") is not None:
        lines.append(f"   On {who}: {_pct(mm['champ_win_rate'])} ({_pct(mm.get('champ_diff'), True)} vs without, "
                     f"{mm.get('champ_games', 0):,} games, picked {_pct(mm.get('champ_pick_rate'))})")
    else:
        lines.append(f"   On {who}: no games")
    if mm.get("synergy"):
        lines.append("   Synergy: " + "  ·  ".join(
            f"{s['partner']} {_pct(s['synergy'], True)} ({_pct(s['win_rate'])})" for s in mm["synergy"]))
    if mm.get("slots"):
        cur = mm.get("current_slot")
        lines.append("   By slot: " + "  ".join(
            (f"▶{s['slot']}: {_pct(s['win_rate'])}" if s["slot"] == cur else f"{s['slot']}: {_pct(s['win_rate'])}")
            for s in mm["slots"]))
    return lines


TABLE_SEP = "│"   # the overlay shows lines containing this in a monospace font
# Column widths: name, tier, win, pick, slot, synergy. Total ~57 chars fits the 440px overlay in
# Consolas 9pt (~7px per char).
_WIDTHS = (17, 4, 6, 6, 4, 4)


def _table_row(cells: list[str]) -> str:
    name, *rest = cells
    w = _WIDTHS[0]
    name = name if len(name) <= w else name[:w - 1] + "…"
    return f" {name:<{w}} {TABLE_SEP} " + f" {TABLE_SEP} ".join(
        f"{c:>{width}}" for c, width in zip(rest, _WIDTHS[1:]))


def _table_rule() -> str:
    return "─" * (_WIDTHS[0] + 2) + "┼" + "┼".join("─" * (w + 2) for w in _WIDTHS[1:])


def diff_label(diff: float | None) -> str:
    """A win-rate difference in whole points rounded toward zero: 5.7 -> "+5%", -5.7 -> "-5%",
    0.4 -> "0%", None -> "?"."""
    if diff is None:
        return "?"
    points = int(diff)  # truncates toward zero, so a gap is never overstated
    return f"{points:+d}%" if points else "0%"


def synergy_label(mm: dict, owned: list[str]) -> str:
    """Total synergy with the augments already owned. "-" when nothing is owned yet, "?" when
    there is no pair data with any owned augment (unknown, not bad)."""
    if not owned:
        return "-"
    pairs = mm.get("synergy") or []
    return diff_label(sum(p["synergy"] for p in pairs if p["synergy"] is not None)) if pairs else "?"


def _card_row(c: dict, slot: int, owned: list[str]) -> tuple[list[str], bool]:
    """[name, tier, win rate, pick rate, slot, synergy] for one card, and whether any cell fell back
    to the all-champion number (marked with *).
      tier:      op.gg tier on this champion, else mayhemmeta overall tier
      win rate:  win rate on this champion, else overall win rate
      pick rate: pick rate on this champion, else overall pick rate
      slot:      current slot's win rate vs the augment's average, e.g. "+5%" (all champions)
      synergy:   summed synergy with the owned augments, e.g. "+4%" (all champions)"""
    mm = c.get("mm") or {}
    fallback = False

    def pick(primary, overall, fmt):
        nonlocal fallback
        if primary is not None:
            return fmt(primary)
        if overall is not None:
            fallback = True
            return fmt(overall) + "*"
        return "–"

    slot_diff = next((s["diff"] for s in mm.get("slots", []) if s["slot"] == slot), None)
    tier = pick(tier_letter(c["tier"]) if c["tier"] is not None else None, mm.get("tier"), str)
    win = pick(mm.get("champ_win_rate"), mm.get("win_rate"), lambda v: f"{v:.1f}%")
    pick_rate = pick(mm.get("champ_pick_rate"), mm.get("pick_rate"), lambda v: f"{v:.1f}%")
    return [c["name"], tier, win, pick_rate, diff_label(slot_diff), synergy_label(mm, owned)], fallback


def format_ranking(cards: list[dict], game: LiveGame | None, owned: list[str] | None = None) -> str:
    if not cards:
        return "No known augment cards detected. Press F8 when the cards are on screen."
    who = game.my_champion if game else "unknown champion"
    owned = owned or []
    slot = min(len(owned) + 1, 4)
    rows = [_card_row(c, slot, owned) for c in cards]
    lines = [f"{who}  ·  augment slot {slot}", "",
             _table_row(["Augment", "Tier", "Win", "Pick", "Slot", "Syn"]), _table_rule(),
             *(_table_row(cells) for cells, _ in rows), ""]
    if any(fb for _, fb in rows):
        lines.append(f"* overall: no {who} data")
    lines.append(f"Slot: win rate in slot {slot} vs the augment's average")
    lines.append("Syn: extra win rate with " + (", ".join(owned) if owned else "owned augments (none yet)")
                 + "  ? = no data")
    return "\n".join(lines) + "\n\n— AI recommendation —\n"


_ADVICE_SYSTEM = (
    "You are a League of Legends ARAM Mayhem augment coach. The strongest evidence is the card's win rate "
    "on this champion and its change vs games without it (needs a few hundred games to trust), then synergy "
    "with augments already owned (+/- already corrects for each augment's own strength), then the op.gg "
    "tier and overall win rate. Win rate by slot tells you whether it is better early or late (▶ marks the "
    "current slot). Missing synergy data means unknown, not bad. Only go against the numbers with a "
    "concrete reason such as items or the enemy team. Be brief: a one-line verdict naming the pick, then "
    "one short line per card. Under 120 words."
)


def _advice_prompt(cards: list[dict], game: LiveGame | None, owned: list[str]) -> str:
    lines = [f"Augments already chosen: {', '.join(owned) or 'none yet'}"]
    if game:
        lines += [f"Champion: {game.my_champion} (level {game.level})",
                  f"Items: {', '.join(i for i in game.my_items if i) or 'none'}",
                  f"Allies: {', '.join(game.allies)}", f"Enemies: {', '.join(game.enemies)}"]
    lines.append("Cards on offer, in on-screen order (left to right):")
    who = game.my_champion if game else "this champion"
    for i, c in enumerate(cards, 1):
        grade = "no data" if c["tier"] is None else f"{tier_letter(c['tier'])} tier"
        lines.append(f"{i}. {c['name']} [op.gg {grade}]: {c['desc']}")
        lines += card_stat_lines(c, who)
    return "\n".join(lines)


def stream_augment_advice(cards: list[dict], game: LiveGame | None, owned: list[str] | None = None) -> Iterator[str]:
    """Yields the recommendation in chunks as the model writes it. Text only: the card names and
    descriptions come from the database, so no image is needed."""
    stream = chat(
        model=AUGMENT_MODEL,
        keep_alive=KEEP_ALIVE,
        messages=[
            {"role": "system", "content": _ADVICE_SYSTEM},
            {"role": "user", "content": _advice_prompt(cards, game, owned or [])},
        ],
        stream=True,
    )
    for part in stream:
        if part.message.content:
            yield part.message.content
