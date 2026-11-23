"""Collect match-v5 data into SQLite and compute champion / augment-on-champion win rates.

    python -m lolai.integrations.riot.opgg.stats probe <riotName> <tag>     # inspect one match's raw fields
    python -m lolai.integrations.riot.opgg.stats collect 2000               # fetch ~2000 matches
    python -m lolai.integrations.riot.opgg.stats report [patch]             # print win rates
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from collections import deque

from lolai.integrations.riot.db import DB_PATH
from lolai.integrations.riot.web_api import ARAM_QUEUE_ID, RiotApi

_AUGMENT_KEY = re.compile(r"^playerAugment\d+$")  # Arena uses this; verify with `probe` for other modes

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (match_id TEXT PRIMARY KEY, patch TEXT, queue INTEGER);
CREATE TABLE IF NOT EXISTS picks (match_id TEXT, puuid TEXT, champion TEXT, win INTEGER,
                                  PRIMARY KEY (match_id, puuid));
CREATE TABLE IF NOT EXISTS match_augments (match_id TEXT, puuid TEXT, champion TEXT, augment INTEGER, win INTEGER);
CREATE INDEX IF NOT EXISTS ix_match_aug ON match_augments (champion, augment);
"""


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def _patch(game_version: str) -> str:
    return ".".join(game_version.split(".")[:2])  # "15.4.612.3" -> "15.4"


def probe(api: RiotApi, game_name: str, tag: str) -> None:
    """Print the participant fields of one recent match: confirms whether augments are present."""
    puuid = api.puuid(game_name, tag)
    ids = api.match_ids(puuid, queue=ARAM_QUEUE_ID, count=1) if puuid else []
    if not ids:
        print("No ARAM match found for that player.")
        return
    info = api.match(ids[0])["info"]
    p = info["participants"][0]
    print("queueId:", info["queueId"], "| gameVersion:", info["gameVersion"])
    print("augment-like keys:", [k for k in p if "ugment" in k] or "NONE")


def _store_match(con: sqlite3.Connection, match: dict) -> list[str]:
    info, meta = match["info"], match["metadata"]
    con.execute("INSERT OR IGNORE INTO matches VALUES (?,?,?)",
                (meta["matchId"], _patch(info["gameVersion"]), info["queueId"]))
    for p in info["participants"]:
        win = int(p["win"])
        con.execute("INSERT OR IGNORE INTO picks VALUES (?,?,?,?)",
                    (meta["matchId"], p["puuid"], p["championName"], win))
        for k, v in p.items():
            if _AUGMENT_KEY.match(k) and v:
                con.execute("INSERT INTO match_augments VALUES (?,?,?,?,?)",
                            (meta["matchId"], p["puuid"], p["championName"], v, win))
    con.commit()
    return meta["participants"]


def collect(api: RiotApi, max_matches: int, queue: int = ARAM_QUEUE_ID, tier_queue: str = "RANKED_SOLO_5x5") -> None:
    """Seed with challenger players, then snowball through the players found in each match."""
    con = _db()
    seen = {r[0] for r in con.execute("SELECT match_id FROM matches")}
    league = api.call("challenger_league", queue=tier_queue) or {}
    players = deque(e["puuid"] for e in league.get("entries", []) if e.get("puuid"))
    visited: set[str] = set()
    fetched = 0
    while players and fetched < max_matches:
        puuid = players.popleft()
        if puuid in visited:
            continue
        visited.add(puuid)
        for match_id in api.match_ids(puuid, queue=queue, count=100):
            if match_id in seen:
                continue
            match = api.match(match_id)
            if not match:
                continue
            seen.add(match_id)
            players.extend(p for p in _store_match(con, match) if p not in visited)
            fetched += 1
            if fetched % 50 == 0:
                print(f"{fetched}/{max_matches} matches")
            if fetched >= max_matches:
                break
    print(f"Done: {fetched} new matches -> {DB_PATH}")


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the win-rate confidence interval: ranks small samples fairly."""
    if n == 0:
        return 0.0
    p = wins / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def champion_winrates(con: sqlite3.Connection, patch: str, min_games: int = 100):
    rows = con.execute(
        """SELECT p.champion, COUNT(*), SUM(p.win) FROM picks p JOIN matches m USING (match_id)
           WHERE m.patch = ? GROUP BY p.champion HAVING COUNT(*) >= ?""", (patch, min_games)).fetchall()
    return sorted(((c, n, w / n, wilson_lower(w, n)) for c, n, w in rows), key=lambda r: -r[3])


def augment_winrates(con: sqlite3.Connection, patch: str, champion: str | None = None, min_games: int = 30):
    """Win rate per (champion, augment); pass champion to see the best augments for one champion."""
    sql = """SELECT a.champion, a.augment, COUNT(*), SUM(a.win) FROM match_augments a JOIN matches m USING (match_id)
             WHERE m.patch = ?"""
    args: list = [patch]
    if champion:
        sql += " AND a.champion = ?"
        args.append(champion)
    sql += " GROUP BY a.champion, a.augment HAVING COUNT(*) >= ?"
    args.append(min_games)
    return sorted(((c, a, n, w / n, wilson_lower(w, n)) for c, a, n, w in con.execute(sql, args)),
                  key=lambda r: -r[4])


def _report(patch: str | None) -> None:
    con = _db()
    patch = patch or con.execute("SELECT patch FROM matches ORDER BY patch DESC LIMIT 1").fetchone()[0]
    print(f"== Champions, patch {patch} ==")
    for c, n, wr, score in champion_winrates(con, patch)[:20]:
        print(f"{c:<16} {n:>5} games  {wr:.1%}  (score {score:.3f})")
    print(f"\n== Augment x champion, patch {patch} ==")
    for c, a, n, wr, score in augment_winrates(con, patch)[:20]:
        print(f"{c:<16} augment {a:<6} {n:>4} games  {wr:.1%}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["report"]
    if cmd == "probe":
        probe(RiotApi(), args[0], args[1])
    elif cmd == "collect":
        collect(RiotApi(), int(args[0]) if args else 500)
    else:
        _report(args[0] if args else None)
