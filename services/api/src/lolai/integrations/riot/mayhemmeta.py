"""Load ARAM Mayhem augment win rates from mayhemmeta.com into SQLite (data/stats.sqlite).

Tables (all keyed by patch, e.g. "16.23"):

  mm_augments(id PK, name, rarity, icon_url)
      What each augment is. rarity: "Silver" | "Gold" | "Prismatic".

  mm_augment_stats(patch, augment_id, tier, win_rate, pick_rate, games, versatility)
      Overall augment win rate across all champions (the /augments leaderboard).

  mm_augment_champion(patch, augment_id, champion_id, champion, games, win_rate, pick_rate,
                      baseline_win_rate)
      Augment win rate on one champion. baseline_win_rate = that champion's win rate in games
      where they did NOT take the augment, so win_rate - baseline_win_rate is the augment's effect.

  mm_augment_pairs(patch, augment_id, partner_id, pick_rate, win_rate, synergy)
      Win rate when both augments are taken together. synergy = win_rate minus an expected
      rate that already accounts for both augments' solo strength (mayhemmeta's "(+/-)").
      Only the site's top lists are published: per augment, the 10 best and 10 most frequent
      partners with 100+ games, so this is NOT every pair.

  mm_augment_slots(patch, augment_id, slot, level, share, win_rate, diff)
      Win rate when the augment is taken in slot 1-4 (levels 1/7/11/15), all champions.
      share = % of this augment's picks made in that slot; diff = win_rate minus the augment's average.

    python -m lolai.integrations.riot.mayhemmeta load              # leaderboard + every augment page
    python -m lolai.integrations.riot.mayhemmeta load 1095 1134    # just these augment ids
    python -m lolai.integrations.riot.mayhemmeta show 1095         # print what is stored for one augment

The pages are Next.js server components: the data is in the self.__next_f "flight" stream, as a
React element tree whose rows may be split into lazy "$Lxx" references. We parse that stream
rather than the HTML.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from typing import Any, Iterator

import requests

from lolai.integrations.riot.db import DB_PATH

BASE = "https://mayhemmeta.com"
_HEADERS = {"User-Agent": "Mozilla/5.0 (personal lolai tool)"}
REQUEST_DELAY = 1.0  # seconds between pages; be polite

SCHEMA = """
CREATE TABLE IF NOT EXISTS mm_augments (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    rarity    TEXT,
    icon_url  TEXT
);
CREATE TABLE IF NOT EXISTS mm_augment_stats (
    patch       TEXT NOT NULL,
    augment_id  INTEGER NOT NULL REFERENCES mm_augments(id),
    tier        TEXT,
    win_rate    REAL,
    pick_rate   REAL,
    games       INTEGER,
    versatility REAL,
    PRIMARY KEY (patch, augment_id)
);
CREATE TABLE IF NOT EXISTS mm_augment_champion (
    patch             TEXT NOT NULL,
    augment_id        INTEGER NOT NULL REFERENCES mm_augments(id),
    champion_id       INTEGER NOT NULL,
    champion          TEXT NOT NULL,
    games             INTEGER,
    win_rate          REAL,
    pick_rate         REAL,
    baseline_win_rate REAL,
    PRIMARY KEY (patch, augment_id, champion_id)
);
CREATE INDEX IF NOT EXISTS ix_mm_champ ON mm_augment_champion (patch, champion);
CREATE TABLE IF NOT EXISTS mm_augment_pairs (
    patch       TEXT NOT NULL,
    augment_id  INTEGER NOT NULL REFERENCES mm_augments(id),
    partner_id  INTEGER NOT NULL REFERENCES mm_augments(id),
    pick_rate   REAL,
    win_rate    REAL,
    synergy     REAL,
    PRIMARY KEY (patch, augment_id, partner_id)
);
CREATE TABLE IF NOT EXISTS mm_augment_slots (
    patch       TEXT NOT NULL,
    augment_id  INTEGER NOT NULL REFERENCES mm_augments(id),
    slot        INTEGER NOT NULL,
    level       INTEGER,
    share       REAL,
    win_rate    REAL,
    diff        REAL,
    PRIMARY KEY (patch, augment_id, slot)
);
"""


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


# ---- fetching ---------------------------------------------------------------------------

def _get(url: str) -> str:
    for attempt in range(3):
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"mayhemmeta kept failing for {url}")


# ---- React Server Component stream parsing ------------------------------------------------

def _flight(html: str) -> str:
    """Concatenate the self.__next_f.push([1,"..."]) chunks into the raw RSC stream."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S)
    return "".join(json.loads(f'"{c}"') for c in chunks)


def _tree(html: str) -> tuple[str, list]:
    """Returns (raw stream, list of every stream row with "$Lxx" references resolved)."""
    stream = _flight(html)
    decoder = json.JSONDecoder()
    refs: dict[str, Any] = {}
    for m in re.finditer(r'(?:^|\n)([0-9a-f]+):(?=[\[{"])', stream):
        try:
            refs[m.group(1)] = decoder.raw_decode(stream, m.end())[0]
        except ValueError:
            continue

    def resolve(v: Any, depth: int = 0) -> Any:
        if isinstance(v, str):
            m = re.fullmatch(r"\$L?([0-9a-f]+)", v)
            if m and m.group(1) in refs and depth < 50:
                return resolve(refs[m.group(1)], depth + 1)
            return v
        if isinstance(v, list):
            return [resolve(x, depth) for x in v]
        if isinstance(v, dict):
            return {k: resolve(x, depth) for k, x in v.items()}
        return v

    return stream, [resolve(v) for v in refs.values()]


def _is_element(node: Any) -> bool:
    return isinstance(node, list) and len(node) == 4 and node[0] == "$" and isinstance(node[3], dict)


def _elements(node: Any, tag: str | None = None) -> Iterator[list]:
    """Every React element ["$", tag, key, props] under node, depth first."""
    if _is_element(node):
        if tag is None or node[1] == tag:
            yield node
        yield from _elements(node[3].get("children"), tag)
    elif isinstance(node, list):
        for x in node:
            yield from _elements(x, tag)
    elif isinstance(node, dict):
        for x in node.values():
            yield from _elements(x, tag)


def _text(node: Any) -> str:
    if isinstance(node, (str, int, float)) and not isinstance(node, bool):
        return str(node)
    if _is_element(node):
        return _text(node[3].get("children"))
    if isinstance(node, list):
        return "".join(_text(x) for x in node)
    return ""


def _rate_and_diff(s: str) -> tuple[float | None, float | None]:
    """"69.51%(+10.43)" -> (69.51, 10.43)."""
    m = re.match(r"\s*([\d.]+)%\s*\(([+-])\s*([\d.]+)\)", s)
    if not m:
        return _num(s), None
    return float(m.group(1)), float(m.group(3)) * (-1 if m.group(2) == "-" else 1)


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _id_rows(tree: list) -> dict[int, list]:
    """<tr key="<augment id>"> elements, deduplicated (resolved refs repeat subtrees)."""
    return {int(e[2]): e for e in _elements(tree, "tr") if isinstance(e[2], str) and e[2].isdigit()}


# ---- page parsers -----------------------------------------------------------------------

def parse_leaderboard(html: str) -> tuple[str, list[dict]]:
    """Returns (patch, rows) from /augments."""
    stream, tree = _tree(html)
    m = re.search(r'"defaultPatch":"([\d.]+)"', stream)
    patch = m.group(1) if m else "unknown"
    rows = []
    for aug_id, tr in _id_rows(tree).items():
        tds = list(_elements(tr[3]["children"], "td"))
        cells = {td[2]: _text(td) for td in tds if td[2]}
        if "win_rate" not in cells:
            continue
        rows.append({
            "augment_id": aug_id,
            "name": _text(tds[1]) if len(tds) > 1 else "",
            "tier": _text(tds[2]) if len(tds) > 2 else None,
            "win_rate": _num(cells["win_rate"]),
            "pick_rate": _num(cells.get("pick_rate", "")),
            "games": int(_num(cells.get("games", "")) or 0),
            "versatility": _num(cells.get("versatility_score", "")),
        })
    return patch, rows


def _json_arrays(stream: str, key: str) -> Iterator[list]:
    decoder = json.JSONDecoder()
    for m in re.finditer(rf'"{key}":\[', stream):
        try:
            yield decoder.raw_decode(stream, m.end() - 1)[0]
        except ValueError:
            continue


def parse_augment_page(html: str) -> dict:
    """Returns {"augments": [catalog], "champions": [...], "pairs": [...]} from /augments/<id>."""
    stream, tree = _tree(html)

    catalog = {}
    for arr in _json_arrays(stream, "augments"):
        for a in arr:
            if isinstance(a, dict) and "augment_id" in a:
                catalog[a["augment_id"]] = a

    champions: dict[int, dict] = {}
    for arr in _json_arrays(stream, "rows"):
        for r in arr:
            if isinstance(r, dict) and "baseline_win_rate" in r and r.get("games"):
                champions[r["id"]] = {
                    "champion_id": r["id"],
                    "champion": r["name"],
                    "games": r["games"],
                    "win_rate": _num(r["win_rate"]),
                    "pick_rate": _num(r["pick_rate"]),
                    "baseline_win_rate": _num(r["baseline_win_rate"]),
                }

    pairs: dict[int, dict] = {}
    slots: dict[int, dict] = {}
    for section in _elements(tree, "section"):
        heads = [_text(h) for h in _elements(section, "h2")]
        if any("Co-picked" in h for h in heads):
            for partner_id, tr in _id_rows(section).items():
                tds = [_text(td) for td in _elements(tr[3]["children"], "td")]
                if len(tds) < 3:
                    continue
                win_rate, synergy = _rate_and_diff(tds[2])
                pairs[partner_id] = {"partner_id": partner_id, "pick_rate": _num(tds[1]),
                                     "win_rate": win_rate, "synergy": synergy}
        elif any("by Slot" in h for h in heads):
            for slot, tr in _id_rows(section).items():   # <tr key="1".."4">
                tds = [_text(td) for td in _elements(tr[3]["children"], "td")]
                if len(tds) < 3:
                    continue
                level = re.search(r"Lvl\.\s*(\d+)", tds[0])
                win_rate, diff = _rate_and_diff(tds[2])
                slots[slot] = {"slot": slot, "level": int(level.group(1)) if level else None,
                               "share": _num(tds[1]), "win_rate": win_rate, "diff": diff}

    return {"augments": list(catalog.values()), "champions": list(champions.values()),
            "pairs": list(pairs.values()), "slots": list(slots.values())}


# ---- loading ------------------------------------------------------------------------------

def load(augment_ids: list[int] | None = None) -> None:
    con = connect()
    patch, board = parse_leaderboard(_get(f"{BASE}/augments"))
    print(f"patch {patch}: {len(board)} augments on the leaderboard")
    for r in board:
        con.execute("INSERT INTO mm_augments (id, name) VALUES (?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET name = excluded.name", (r["augment_id"], r["name"]))
        con.execute("INSERT OR REPLACE INTO mm_augment_stats VALUES (?,?,?,?,?,?,?)",
                    (patch, r["augment_id"], r["tier"], r["win_rate"], r["pick_rate"], r["games"],
                     r["versatility"]))
    con.commit()

    ids = augment_ids or [r["augment_id"] for r in board]
    for n, aug_id in enumerate(ids, 1):
        time.sleep(REQUEST_DELAY)
        try:
            page = parse_augment_page(_get(f"{BASE}/augments/{aug_id}"))
        except requests.HTTPError as error:
            print(f"  {aug_id}: {error}")
            continue
        for a in page["augments"]:
            con.execute(
                "INSERT INTO mm_augments VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "name = excluded.name, rarity = excluded.rarity, icon_url = excluded.icon_url",
                (a["augment_id"], a["name"], a.get("rarity"), a.get("icon_url")))
        con.execute("DELETE FROM mm_augment_champion WHERE patch = ? AND augment_id = ?", (patch, aug_id))
        con.executemany("INSERT INTO mm_augment_champion VALUES (?,?,?,?,?,?,?,?)", [
            (patch, aug_id, c["champion_id"], c["champion"], c["games"], c["win_rate"], c["pick_rate"],
             c["baseline_win_rate"]) for c in page["champions"]])
        con.execute("DELETE FROM mm_augment_pairs WHERE patch = ? AND augment_id = ?", (patch, aug_id))
        con.executemany("INSERT INTO mm_augment_pairs VALUES (?,?,?,?,?,?)", [
            (patch, aug_id, p["partner_id"], p["pick_rate"], p["win_rate"], p["synergy"])
            for p in page["pairs"]])
        con.execute("DELETE FROM mm_augment_slots WHERE patch = ? AND augment_id = ?", (patch, aug_id))
        con.executemany("INSERT INTO mm_augment_slots VALUES (?,?,?,?,?,?,?)", [
            (patch, aug_id, s["slot"], s["level"], s["share"], s["win_rate"], s["diff"])
            for s in page["slots"]])
        con.commit()
        print(f"  [{n}/{len(ids)}] {aug_id}: {len(page['champions'])} champions, "
              f"{len(page['pairs'])} pairs, {len(page['slots'])} slots")
    con.close()


# ---- lookups for the advisor ----------------------------------------------------------------

def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def latest_patch(con: sqlite3.Connection) -> str | None:
    rows = [r[0] for r in con.execute("SELECT DISTINCT patch FROM mm_augment_stats")]
    return max(rows, key=lambda p: tuple(int(x) for x in p.split(".") if x.isdigit()), default=None)


def card_stats(con: sqlite3.Connection, names: list[str], champion: str | None,
               owned: list[str] | None = None, patch: str | None = None) -> dict[str, dict]:
    """Display stats for each card name (matched case/punctuation-insensitively), keyed by the name
    as given. Missing data is None / empty, never an error. Keys:
      tier, win_rate, pick_rate, games            overall (all champions)
      champ_games, champ_win_rate, champ_pick_rate, champ_diff   on `champion` (diff vs without it)
      synergy: [{partner, win_rate, synergy}]      with each owned augment that has pair data
      slots: [{slot, level, share, win_rate, diff}], current_slot (1-4, from len(owned))
    """
    owned = owned or []
    patch = patch or latest_patch(con)
    ids = {normalize(n): i for i, n in con.execute("SELECT id, name FROM mm_augments")}
    owned_ids = {ids[normalize(o)]: o for o in owned if normalize(o) in ids}
    champ_norm = normalize(champion or "")
    out: dict[str, dict] = {}
    for name in names:
        aug_id = ids.get(normalize(name))
        s: dict = {"patch": patch, "current_slot": min(len(owned) + 1, 4), "synergy": [], "slots": []}
        out[name] = s
        if aug_id is None or patch is None:
            continue
        row = con.execute("SELECT tier, win_rate, pick_rate, games FROM mm_augment_stats "
                          "WHERE patch = ? AND augment_id = ?", (patch, aug_id)).fetchone()
        if row:
            s.update(tier=row[0], win_rate=row[1], pick_rate=row[2], games=row[3])
        for champ, games, wr, pr, base in con.execute(
                "SELECT champion, games, win_rate, pick_rate, baseline_win_rate FROM mm_augment_champion "
                "WHERE patch = ? AND augment_id = ?", (patch, aug_id)):
            if normalize(champ) == champ_norm:
                s.update(champ_games=games, champ_win_rate=wr, champ_pick_rate=pr,
                         champ_diff=round(wr - base, 2) if wr is not None and base is not None else None)
                break
        for partner_id, partner in owned_ids.items():
            pair = con.execute(
                "SELECT win_rate, synergy FROM mm_augment_pairs WHERE patch = ? AND "
                "((augment_id = ? AND partner_id = ?) OR (augment_id = ? AND partner_id = ?))",
                (patch, aug_id, partner_id, partner_id, aug_id)).fetchone()
            if pair:
                s["synergy"].append({"partner": partner, "win_rate": pair[0], "synergy": pair[1]})
        s["slots"] = [dict(zip(("slot", "level", "share", "win_rate", "diff"), r)) for r in con.execute(
            "SELECT slot, level, share, win_rate, diff FROM mm_augment_slots "
            "WHERE patch = ? AND augment_id = ? ORDER BY slot", (patch, aug_id))]
    return out


def show(aug_id: int) -> None:
    con = connect()
    row = con.execute(
        "SELECT a.name, a.rarity, s.patch, s.tier, s.win_rate, s.pick_rate, s.games FROM mm_augments a "
        "JOIN mm_augment_stats s ON s.augment_id = a.id WHERE a.id = ? ORDER BY s.patch DESC LIMIT 1",
        (aug_id,)).fetchone()
    if not row:
        print("not loaded")
        return
    name, rarity, patch, tier, wr, pr, games = row
    print(f"{name} ({rarity}) patch {patch}: tier {tier}, {wr}% WR, {pr}% PR, {games} games")
    print("\nBest champions (min 100 games):")
    for c in con.execute(
            "SELECT champion, games, win_rate, win_rate - baseline_win_rate FROM mm_augment_champion "
            "WHERE patch = ? AND augment_id = ? AND games >= 100 ORDER BY win_rate DESC LIMIT 10",
            (patch, aug_id)):
        print(f"  {c[0]:<14} {c[2]:6.2f}%  ({c[3]:+.2f} vs without)  {c[1]} games")
    print("\nCombos:")
    for p in con.execute(
            "SELECT a.name, p.win_rate, p.synergy, p.pick_rate FROM mm_augment_pairs p "
            "JOIN mm_augments a ON a.id = p.partner_id WHERE p.patch = ? AND p.augment_id = ? "
            "ORDER BY p.synergy DESC", (patch, aug_id)):
        print(f"  {p[0]:<28} {p[1]:6.2f}%  (synergy {p[2]:+.2f})  co-pick {p[3]}%")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "load"), sys.argv[2:]
    if cmd == "load":
        load([int(a) for a in args] or None)
    elif cmd == "show":
        show(int(args[0]))
    else:
        print(__doc__)
