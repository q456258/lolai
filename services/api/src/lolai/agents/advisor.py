"""Turns game state into advice: best champion in champ select, best augment on screen."""

from __future__ import annotations

from ollama import chat
from pydantic import BaseModel

from lolai.domain.enums.model import Model
from lolai.integrations.riot.models import ChampSelect, LiveGame
from lolai.integrations.riot.opgg.opgg import connect, match_name


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

class AugmentOption(BaseModel):
    name: str
    score: int  # 1-10 for this champion and matchup
    pros: str
    cons: str


class AugmentAdvice(BaseModel):
    options: list[AugmentOption]
    best: str
    reason: str


_AUGMENT_SYSTEM = (
    "You are a League of Legends augment (Arena/ARAM Mayhem) expert. The image is a screenshot "
    "showing augment cards to choose from. Read each card's name and text exactly, then rate each "
    "for the player's champion, items and the teams. If no augment cards are visible, return an "
    "empty options list and best = ''."
)


def _augment_prompt(game: LiveGame | None, owned: list[str]) -> str:
    chosen = f"Augments I already chose (consider synergy): {', '.join(owned) or 'none yet'}\n"
    if game is None:
        return chosen + "Game context unavailable. Pick the generally strongest augment."
    return chosen + (
        f"My champion: {game.my_champion} (level {game.level})\n"
        f"My items: {', '.join(i for i in game.my_items if i) or 'none'}\n"
        f"Allies: {', '.join(game.allies)}\n"
        f"Enemies: {', '.join(game.enemies)}"
    )


def advise_augments(screenshot: bytes, game: LiveGame | None, owned: list[str] | None = None) -> AugmentAdvice:
    res = chat(
        model=Model.QWEN_VL.value,
        messages=[
            {"role": "system", "content": _AUGMENT_SYSTEM},
            {"role": "user", "content": _augment_prompt(game, owned or []), "images": [screenshot]},
        ],
        format=AugmentAdvice.model_json_schema(),
    )
    return _verify_augments(AugmentAdvice.model_validate_json(res.message.content))


def _verify_augments(advice: AugmentAdvice) -> AugmentAdvice:
    """Keep only cards whose name matches an augment in the database (name or name_norm).

    Anything else the vision model "read" is not an augment (or was misread) and is dropped."""
    con = connect()
    try:
        if not con.execute("SELECT 1 FROM augments LIMIT 1").fetchone():
            raise RuntimeError("Augment database is empty. Run: python -m lolai.integrations.riot.opgg.opgg load")
        kept: dict[str, AugmentOption] = {}
        for o in advice.options:
            name = match_name(con, o.name)
            if name and name not in kept:
                kept[name] = o.model_copy(update={"name": name})
        best = match_name(con, advice.best)
    finally:
        con.close()
    if not kept:
        return AugmentAdvice(options=[], best="", reason="")
    if best not in kept:  # the model's pick was dropped: fall back to the top-scored verified card
        top = max(kept.values(), key=lambda o: o.score)
        return AugmentAdvice(options=list(kept.values()), best=top.name, reason=top.pros)
    return AugmentAdvice(options=list(kept.values()), best=best, reason=advice.reason)


def format_augments(a: AugmentAdvice) -> str:
    if not a.options:
        return "No known augment cards detected. Press F8 when the cards are on screen."
    lines = [f"★ PICK: {a.best}", f"  {a.reason}", ""]
    for o in sorted(a.options, key=lambda o: -o.score):
        lines += [f"{o.name}  [{o.score}/10]", f"  + {o.pros}", f"  - {o.cons}", ""]
    return "\n".join(lines).rstrip()
