"""Turns game state into advice: best champion in champ select, best augment on screen."""

from __future__ import annotations

from collections.abc import Iterator

from ollama import chat
from pydantic import BaseModel

from lolai.domain.enums.model import Model
from lolai.integrations.riot.models import ChampSelect, LiveGame
from lolai.integrations.riot.champions import champ_slug
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


def rank_augments(names: list[str], game: LiveGame | None) -> list[dict]:
    """Keep only names that match a known augment (exact, or fuzzy for misreads), then rank them
    for the player's champion from the database. Unknown text is not an augment and is dropped."""
    con = connect()
    try:
        if not con.execute("SELECT 1 FROM augments LIMIT 1").fetchone():
            raise RuntimeError("Augment database is empty. Run: python -m lolai.integrations.riot.opgg.opgg augments")
        verified = list(dict.fromkeys(n for n in (match_name(con, x) for x in names) if n))
        slug = champ_slug(game.my_champion) if game else None
        return rank_cards(con, slug, verified)
    finally:
        con.close()


TIER_LETTERS = "SABCDE"  # op.gg tier 0..5


def tier_letter(tier: int | None) -> str:
    if tier is None:
        return "?"
    return TIER_LETTERS[tier] if 0 <= tier < len(TIER_LETTERS) else str(tier)


def format_ranking(cards: list[dict], game: LiveGame | None) -> str:
    if not cards:
        return "No known augment cards detected. Press F8 when the cards are on screen."
    who = game.my_champion if game else "unknown champion"
    lines = [f"Augment tiers for {who}  (left to right)", ""]
    for i, c in enumerate(cards, 1):
        grade = "no data for this champion" if c["tier"] is None else f"{tier_letter(c['tier'])} tier"
        lines.append(f"{i}. {c['name']}  -  {grade}")
    return "\n".join(lines) + "\n\n— AI recommendation —\n"


_ADVICE_SYSTEM = (
    "You are a League of Legends ARAM Mayhem augment coach. The op.gg tier for each card on this champion "
    "(S best, then A, B, C, D, E worst) is a strong prior; only deviate from it with a concrete reason such "
    "as synergy with the player's items or chosen augments, or the enemy team. Be brief: a one-line verdict "
    "naming the pick, then one short line per card. Under 120 words."
)


def _advice_prompt(cards: list[dict], game: LiveGame | None, owned: list[str]) -> str:
    lines = [f"Augments already chosen: {', '.join(owned) or 'none yet'}"]
    if game:
        lines += [f"Champion: {game.my_champion} (level {game.level})",
                  f"Items: {', '.join(i for i in game.my_items if i) or 'none'}",
                  f"Allies: {', '.join(game.allies)}", f"Enemies: {', '.join(game.enemies)}"]
    lines.append("Cards on offer, in on-screen order (left to right):")
    for c in cards:
        grade = "no data" if c["tier"] is None else f"{tier_letter(c['tier'])} tier"
        lines.append(f"{cards.index(c) + 1}. {c['name']} [{grade}]: {c['desc']}")
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
