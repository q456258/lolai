"""Load ARAM Mayhem augment tiers from op.gg into SQLite.

Tables (in the same data/stats.sqlite as stats.py):

  augments(id PK, key, name, name_norm, rarity, description, tooltip, icon_url)
      One row per augment: what it is. Latest text wins when re-run.

  augment_stats(patch, mode, champion_slug, champion, augment_id -> augments.id,
                tier, performance, pick_rate)          PK(patch, mode, champion_slug, augment_id)
      One row per augment per champion per patch: how good it is.
      tier: 0 = best ... 5 = worst/no data (as op.gg orders them)
      performance: op.gg strength index (~100 = average), NOT a win rate
      pick_rate: percent of games

    python -m lolai.integrations.riot.opgg.opgg augments        # full augment list, 1 request
    python -m lolai.integrations.riot.opgg.opgg load            # all champions (~170 requests, 1/sec)
    python -m lolai.integrations.riot.opgg.opgg load teemo ahri # just these slugs
    python -m lolai.integrations.riot.opgg.opgg show teemo      # best augments for a champion
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from difflib import SequenceMatcher

import requests

from lolai.integrations.riot.champions import _load as _load_champions
from lolai.integrations.riot.db import DB_PATH

MODE = "aram-mayhem"
_BASE = f"https://op.gg/lol/modes/{MODE}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (personal lolai tool)"}
REQUEST_DELAY = 1.0  # seconds between pages; be polite

SCHEMA = """
CREATE TABLE IF NOT EXISTS augments (
    id          INTEGER PRIMARY KEY,           -- op.gg / Riot augment id
    key         TEXT UNIQUE,                   -- "ARAM_Recursion"
    name        TEXT NOT NULL,
    name_norm   TEXT NOT NULL,                 -- lowercase, alphanumerics only: for matching screen text
    rarity      INTEGER,                       -- 1 silver, 4 gold, 8 prismatic (inferred); NULL if unknown
    description TEXT,
    tooltip     TEXT,
    icon_url    TEXT
);
CREATE INDEX IF NOT EXISTS ix_augments_norm ON augments (name_norm);

CREATE TABLE IF NOT EXISTS augment_stats (
    patch         TEXT NOT NULL,               -- "16.23"
    mode          TEXT NOT NULL,               -- "aram-mayhem"
    champion_slug TEXT NOT NULL,               -- op.gg slug == lowercase Data Dragon id ("monkeyking")
    champion      TEXT NOT NULL,               -- display name ("Wukong")
    augment_id    INTEGER NOT NULL REFERENCES augments(id),
    tier          INTEGER NOT NULL,
    performance   REAL,
    pick_rate     REAL,
    PRIMARY KEY (patch, mode, champion_slug, augment_id)
);
CREATE INDEX IF NOT EXISTS ix_stats_champ ON augment_stats (patch, mode, champion_slug, tier);
"""


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---- fetching / parsing -----------------------------------------------------------------

def _get(url: str) -> str:
    for attempt in range(3):
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"op.gg kept failing for {url}")


def parse_page(html: str) -> tuple[str, list[dict]]:
    """Returns (patch, augments). The data is embedded in a Next.js flight chunk as JSON."""
    patch = re.search(r"Patch (\d+\.\d+)", html)
    for chunk in re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S):
        text = json.loads(f'"{chunk}"')  # un-escape the JS string
        if '"performance"' in text and '"largeIcon"' in text:
            start = text.index('"data":') + len('"data":')
            augments, _ = json.JSONDecoder().raw_decode(text[start:])
            if patch and augments:
                return patch.group(1), augments
    raise ValueError("Augment data not found: op.gg page structure may have changed")


def champion_slugs() -> list[str]:
    html = _get(f"{_BASE}/teemo/augments")
    return sorted(set(re.findall(rf"/lol/modes/{MODE}/([a-z0-9]+)/(?:build|augments)", html)))


# ---- storing --------------------------------------------------------------------------------

def _upsert_augment(con: sqlite3.Connection, a: dict) -> None:
    con.execute(
        """INSERT INTO augments (id, key, name, name_norm, rarity, description, tooltip, icon_url)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET key=excluded.key, name=excluded.name,
             name_norm=excluded.name_norm, rarity=excluded.rarity,
             description=excluded.description, tooltip=excluded.tooltip, icon_url=excluded.icon_url""",
        (a["id"], a.get("key"), a["name"], normalize(a["name"]), a.get("rarity"),
         a.get("desc"), a.get("tooltip"), a.get("largeIcon")))


def load_augment_list() -> int:
    """Fetch the mode page, which lists every augment in the pool (more than any one champion's page),
    and store their definitions. Returns how many were stored. Per-champion stats come from `load`."""
    patch, augments = parse_page(_get(_BASE))
    named = [a for a in augments if a.get("name")]
    con = connect()
    for a in named:
        _upsert_augment(con, a)
    con.commit()
    print(f"Stored {len(named)} augments from the mode page (patch {patch}); "
          f"{len(augments) - len(named)} unnamed entries skipped.")
    return len(named)


def store(con: sqlite3.Connection, slug: str, patch: str, augments: list[dict]) -> None:
    by_slug = {k.lower(): v for k, v in _load_champions()[1].items()}  # ddragon id -> display name
    champion = by_slug.get(slug, slug)
    # op.gg also returns ids with stats but no name/rarity (tier 5, ~0% pick rate: not in this
    # mode's pool). They can't be matched to anything on screen, so they are skipped.
    for a in (a for a in augments if a.get("name")):
        _upsert_augment(con, a)
        con.execute(
            """INSERT OR REPLACE INTO augment_stats
               (patch, mode, champion_slug, champion, augment_id, tier, performance, pick_rate)
               VALUES (?,?,?,?,?,?,?,?)""",
            (patch, MODE, slug, champion, a["id"], a["tier"], a.get("performance"), a.get("popular")))
    con.commit()


def load(slugs: list[str] | None = None, force: bool = False) -> None:
    con = connect()
    slugs = slugs or champion_slugs()
    for i, slug in enumerate(slugs, 1):
        html = _get(f"{_BASE}/{slug}/augments")
        patch, augments = parse_page(html)
        done = con.execute("SELECT 1 FROM augment_stats WHERE patch=? AND mode=? AND champion_slug=? LIMIT 1",
                           (patch, MODE, slug)).fetchone()
        if done and not force:
            print(f"[{i}/{len(slugs)}] {slug}: already stored for {patch}, skipped")
        else:
            store(con, slug, patch, augments)
            print(f"[{i}/{len(slugs)}] {slug}: {len(augments)} augments (patch {patch})")
        time.sleep(REQUEST_DELAY)


# ---- lookups (what the advisor will call) -------------------------------------------------

def latest_patch(con: sqlite3.Connection) -> str | None:
    row = con.execute("SELECT patch FROM augment_stats WHERE mode=? "
                      "ORDER BY CAST(substr(patch,1,instr(patch,'.')-1) AS INT) DESC, "
                      "CAST(substr(patch,instr(patch,'.')+1) AS INT) DESC LIMIT 1", (MODE,)).fetchone()
    return row[0] if row else None


FUZZY_MIN = 0.85     # similarity (0-1) of normalised names needed to accept a misread
FUZZY_MARGIN = 0.05  # ...and the best candidate must beat the runner-up by this much, else it's ambiguous
# Measured on this data (single-character misreads): 0 wrong matches; near-identical names such as
# "Scoped/Scopier/Scopiest Weapons" are rejected as ambiguous rather than guessed.


def match_name(con: sqlite3.Connection, text: str, fuzzy: bool = True) -> str | None:
    """Canonical augment name for `text`, or None if it isn't a known augment.

    1. exact match on `name` or `name_norm` (case/punctuation-insensitive);
    2. otherwise a fuzzy match, to tolerate misread characters ("Recurs1on"), accepted only when it is
       similar enough and clearly better than every other augment."""
    norm = normalize(text)
    if not norm:
        return None
    row = con.execute("SELECT name FROM augments WHERE name = ? OR name_norm = ?", (text.strip(), norm)).fetchone()
    if row or not fuzzy:
        return row[0] if row else None

    scored = sorted(
        ((SequenceMatcher(None, norm, n).ratio(), name)
         for name, n in con.execute("SELECT name, name_norm FROM augments WHERE length(name_norm) >= 4")),
        reverse=True)
    if not scored or scored[0][0] < FUZZY_MIN:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < FUZZY_MARGIN:
        return None
    return scored[0][1]


def rank_cards(con: sqlite3.Connection, champion_slug: str | None, names: list[str],
               patch: str | None = None) -> list[dict]:
    """Look up on-screen augment cards for a champion, straight from the database (no model involved).

    `names` must be canonical (see match_name). Cards keep the order of `names` (the on-screen order,
    left to right); a card without stats for this champion has tier None. Descriptions have op.gg's
    markup stripped."""
    patch = patch or latest_patch(con)
    cards = []
    for name in names:
        row = con.execute(
            """SELECT a.name, a.rarity, a.description, s.tier, s.performance, s.pick_rate
               FROM augments a LEFT JOIN augment_stats s
                 ON s.augment_id = a.id AND s.patch=? AND s.mode=? AND s.champion_slug=?
               WHERE a.name = ?""", (patch, MODE, champion_slug, name)).fetchone()
        if row:
            cards.append({"name": row[0], "rarity": row[1], "desc": re.sub(r"<[^>]+>", "", row[2] or ""),
                          "tier": row[3], "performance": row[4], "pick_rate": row[5]})
    return cards


def stats_for(con: sqlite3.Connection, champion_slug: str, augment_name: str, patch: str | None = None):
    """Stored stats for one augment on one champion, or None. Matches the name after normalising."""
    patch = patch or latest_patch(con)
    return con.execute(
        """SELECT a.name, a.rarity, s.tier, s.performance, s.pick_rate, a.description
           FROM augment_stats s JOIN augments a ON a.id = s.augment_id
           WHERE s.patch=? AND s.mode=? AND s.champion_slug=? AND a.name_norm=?""",
        (patch, MODE, champion_slug, normalize(augment_name))).fetchone()


def best_for(con: sqlite3.Connection, champion_slug: str, limit: int = 10, patch: str | None = None):
    patch = patch or latest_patch(con)
    return con.execute(
        """SELECT a.name, s.tier, s.performance, s.pick_rate
           FROM augment_stats s JOIN augments a ON a.id = s.augment_id
           WHERE s.patch=? AND s.mode=? AND s.champion_slug=?
           ORDER BY s.tier, s.performance DESC LIMIT ?""",
        (patch, MODE, champion_slug, limit)).fetchall()


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["load"]
    if cmd == "show":
        for row in best_for(connect(), args[0]):
            print(row)
    elif cmd == "augments":
        load_augment_list()
    else:
        load(args or None)
