"""Turns game state into advice: best champion in champ select, best augment on screen."""

from __future__ import annotations

from ollama import chat
from pydantic import BaseModel

from lolai.domain.enums.model import Model
from lolai.integrations.riot.models import ChampSelect, LiveGame


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


def _augment_prompt(game: LiveGame | None) -> str:
    if game is None:
        return "Game context unavailable. Pick the generally strongest augment."
    return (
        f"My champion: {game.my_champion} (level {game.level})\n"
        f"My items: {', '.join(i for i in game.my_items if i) or 'none'}\n"
        f"Allies: {', '.join(game.allies)}\n"
        f"Enemies: {', '.join(game.enemies)}"
    )


def advise_augments(screenshot: bytes, game: LiveGame | None) -> AugmentAdvice:
    res = chat(
        model=Model.QWEN_VL.value,
        messages=[
            {"role": "system", "content": _AUGMENT_SYSTEM},
            {"role": "user", "content": _augment_prompt(game), "images": [screenshot]},
        ],
        format=AugmentAdvice.model_json_schema(),
    )
    return AugmentAdvice.model_validate_json(res.message.content)


def format_augments(a: AugmentAdvice) -> str:
    if not a.options:
        return "No augment cards detected. Press F8 when the cards are on screen."
    lines = [f"★ PICK: {a.best}", f"  {a.reason}", ""]
    for o in sorted(a.options, key=lambda o: -o.score):
        lines += [f"{o.name}  [{o.score}/10]", f"  + {o.pros}", f"  - {o.cons}", ""]
    return "\n".join(lines).rstrip()
