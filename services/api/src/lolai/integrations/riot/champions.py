"""Champion id/key -> display name lookup, backed by Data Dragon (loaded once, lazily)."""

from __future__ import annotations

from functools import lru_cache

import requests

_DDRAGON = "https://ddragon.leagueoflegends.com"


@lru_cache(maxsize=1)
def _load() -> tuple[dict[int, str], dict[str, str]]:
    """Returns (numeric key -> name, ddragon id -> name). Raises on network failure,
    so a failed attempt is not cached and the next call retries."""
    version = requests.get(f"{_DDRAGON}/api/versions.json", timeout=10).json()[0]
    data = requests.get(
        f"{_DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=10
    ).json()["data"]
    by_key = {int(info["key"]): info["name"] for info in data.values()}
    by_id = {champ_id: info["name"] for champ_id, info in data.items()}
    return by_key, by_id


def champ_name(key: int) -> str:
    """Numeric champion key (as used by the LCU), e.g. 103 -> "Ahri"."""
    try:
        return _load()[0].get(key, f"Champion#{key}")
    except requests.RequestException:
        return f"Champion#{key}"


def champ_name_from_id(champ_id: str) -> str:
    """Data Dragon id (as used by the live client), e.g. "MonkeyKing" -> "Wukong"."""
    try:
        return _load()[1].get(champ_id, champ_id)
    except requests.RequestException:
        return champ_id
