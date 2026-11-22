"""Riot Web API (api.riotgames.com) client: the endpoints needed to build champion/augment stats.

Set your key in the environment (e.g. local.env):  RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-...
A personal development key expires every 24h; regenerate it at developer.riotgames.com.
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[6] / "local.env")  # same file orchestrator.py loads

API_KEY_PLACEHOLDER = "RGAPI-your-key-here"  # set the real key as RIOT_API_KEY in local.env, never here

# Personal-key limits: 20 requests / 1s and 100 requests / 2min.
RATE_LIMITS = [(20, 1.0), (100, 120.0)]

ARAM_QUEUE_ID = 450   # Summoner's Rift ARAM; check the queue id of Mayhem/Arena in a real match
ARENA_QUEUE_ID = 1700

# Platform hosts (per-server) vs regional hosts (match-v5, account-v1).
PLATFORM = "na1"        # na1, euw1, kr, ...
REGION = "americas"     # americas, europe, asia, sea

# ---- endpoint templates -----------------------------------------------------------------
# (host kind, path). Fill {placeholders} with str.format.
ENDPOINTS = {
    # account-v1: Riot ID -> puuid
    "account_by_riot_id": ("regional", "/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"),
    # league-v4: seed players from ranked ladders (ARAM has no ladder of its own)
    "challenger_league": ("platform", "/lol/league/v4/challengerleagues/by-queue/{queue}"),
    "league_entries": ("platform", "/lol/league/v4/entries/{queue}/{tier}/{division}"),
    # summoner-v4: puuid -> summoner (only if you need levels/ids)
    "summoner_by_puuid": ("platform", "/lol/summoner/v4/summoners/by-puuid/{puuid}"),
    # match-v5: the data you actually aggregate. params: queue, start, count (<=100), startTime
    "match_ids_by_puuid": ("regional", "/lol/match/v5/matches/by-puuid/{puuid}/ids"),
    "match": ("regional", "/lol/match/v5/matches/{match_id}"),
}

# ---- static data: no API key needed -------------------------------------------------------
DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPIONS = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
COMMUNITY_DRAGON = "https://raw.communitydragon.org/latest/"  # augment definitions live under here


class RiotApiError(RuntimeError):
    pass


class RiotApi:
    def __init__(self, api_key: str | None = None, platform: str = PLATFORM, region: str = REGION) -> None:
        self.api_key = api_key or os.environ.get("RIOT_API_KEY", API_KEY_PLACEHOLDER)
        self.hosts = {
            "platform": f"https://{platform}.api.riotgames.com",
            "regional": f"https://{region}.api.riotgames.com",
        }
        self._http = requests.Session()
        self._sent: list[deque[float]] = [deque() for _ in RATE_LIMITS]

    def _throttle(self) -> None:
        """Block until sending one more request stays inside every rate window."""
        while True:
            now = time.monotonic()
            wait = 0.0
            for (limit, window), sent in zip(RATE_LIMITS, self._sent):
                while sent and now - sent[0] >= window:
                    sent.popleft()
                if len(sent) >= limit:
                    wait = max(wait, window - (now - sent[0]))
            if wait <= 0:
                for sent in self._sent:
                    sent.append(now)
                return
            time.sleep(wait)

    def call(self, name: str, params: dict | None = None, **path_args):
        """Call a named endpoint from ENDPOINTS. Returns parsed JSON, or None on 404."""
        if self.api_key == API_KEY_PLACEHOLDER:
            raise RiotApiError("Set RIOT_API_KEY (https://developer.riotgames.com).")
        host_kind, path = ENDPOINTS[name]
        url = self.hosts[host_kind] + path.format(**{k: quote(str(v), safe="") for k, v in path_args.items()})
        for _ in range(3):
            self._throttle()
            r = self._http.get(url, params=params, headers={"X-Riot-Token": self.api_key}, timeout=10)
            if r.status_code == 429:  # rate limited anyway: honour Retry-After
                time.sleep(float(r.headers.get("Retry-After", 5)))
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise RiotApiError("API key rejected or expired (dev keys last 24h).")
            r.raise_for_status()
            return r.json()
        raise RiotApiError(f"Gave up on {name} after repeated 429s")

    # ---- convenience wrappers -------------------------------------------------------------
    def puuid(self, game_name: str, tag_line: str) -> str | None:
        data = self.call("account_by_riot_id", game_name=game_name, tag_line=tag_line)
        return data["puuid"] if data else None

    def match_ids(self, puuid: str, queue: int = ARAM_QUEUE_ID, start: int = 0, count: int = 100) -> list[str]:
        return self.call("match_ids_by_puuid", params={"queue": queue, "start": start, "count": count},
                         puuid=puuid) or []

    def match(self, match_id: str) -> dict | None:
        return self.call("match", match_id=match_id)
