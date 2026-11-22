"""Reads the in-game state from the Live Client Data API (https://127.0.0.1:2999)."""

from __future__ import annotations

import requests
import urllib3

from lolai.integrations.riot.champions import champ_name_from_id
from lolai.integrations.riot.models import LiveGame

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # self-signed localhost cert

_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"


def _champion(p: dict) -> str:
    # championName is localized; rawChampionName ("game_character_displayname_Ahri") holds the id.
    raw = p.get("rawChampionName", "")
    return champ_name_from_id(raw.rsplit("_", 1)[-1]) if raw else p.get("championName", "?")


def read_live_game() -> LiveGame | None:
    try:
        r = requests.get(_URL, verify=False, timeout=2)
        data = r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None
    if not data or "allPlayers" not in data:
        return None

    active = data.get("activePlayer", {})
    # Riot IDs replaced summoner names; match on whichever the game provides.
    my_ids = {active.get("riotId"), active.get("summonerName")} - {None, ""}
    players = data["allPlayers"]
    me = next((p for p in players if {p.get("riotId"), p.get("summonerName")} & my_ids), None)
    if me is None:
        return None

    return LiveGame(
        my_champion=_champion(me),
        level=int(active.get("level", 0)),
        allies=[_champion(p) for p in players if p.get("team") == me.get("team") and p is not me],
        enemies=[_champion(p) for p in players if p.get("team") != me.get("team")],
        my_items=[i.get("displayName", "") for i in me.get("items", [])],
    )
