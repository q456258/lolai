"""Reads the champ-select lobby from the LCU into a ChampSelect snapshot."""

from __future__ import annotations

from lolai.integrations.riot.champions import champ_name
from lolai.integrations.riot.lcu import Lcu
from lolai.integrations.riot.models import ChampSelect


def _names(ids) -> list[str]:
    return [champ_name(i) for i in ids if i and i > 0]


def _my_action(session: dict, me_cell: int | None) -> str | None:
    """"pick"/"ban" if it is currently my turn to act, else None."""
    for group in session.get("actions", []):
        for a in group:
            if a.get("actorCellId") == me_cell and a.get("isInProgress") and not a.get("completed"):
                return a.get("type")
    return None


def read_champ_select(lcu: Lcu) -> ChampSelect | None:
    s = lcu.get("/lol-champ-select/v1/session")
    if not s:
        return None

    me_cell = s.get("localPlayerCellId")
    mine, position, mates = None, "", []
    for p in s.get("myTeam", []):
        # championId is the locked pick; championPickIntent is the current hover.
        cid = p.get("championId") or p.get("championPickIntent") or 0
        is_me = p.get("cellId") == me_cell
        if is_me:
            position = p.get("assignedPosition", "") or ""
        if cid <= 0:
            continue
        if is_me:
            mine = champ_name(cid)
        else:
            mates.append(champ_name(cid))

    bans = s.get("bans", {})
    my_action = _my_action(s, me_cell)
    pickable = []
    if my_action == "pick":
        pickable = _names(lcu.get("/lol-champ-select/v1/pickable-champion-ids") or [])

    return ChampSelect(
        my_champion=mine,
        my_position=position,
        allies=mates,
        enemies=_names(p.get("championId") for p in s.get("theirTeam", [])),
        bench=_names(c.get("championId") for c in s.get("benchChampions", [])),
        bans=_names([*bans.get("myTeamBans", []), *bans.get("theirTeamBans", [])]),
        pickable=pickable,
        rerolls_left=int(s.get("rerollsRemaining", 0) or 0),
        my_action=my_action,
    )
