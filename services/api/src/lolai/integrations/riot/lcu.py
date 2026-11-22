"""Minimal client for the League Client Update (LCU) local HTTPS API."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import requests
import urllib3

# The LCU serves a self-signed certificate on localhost.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_INSTALL_DIRS = [
    r"C:\Riot Games\League of Legends",
    r"D:\Riot Games\League of Legends",
    "/Applications/League of Legends.app/Contents/LoL",
]
_DISCOVERY_COOLDOWN = 5.0  # seconds between process scans while the client is closed


@dataclass(frozen=True)
class LcuAuth:
    port: int
    password: str


def _from_lockfile() -> LcuAuth | None:
    dirs = [os.environ.get("LEAGUE_DIR", ""), *DEFAULT_INSTALL_DIRS]
    for d in filter(None, dirs):
        try:
            # Format: name:pid:port:password:protocol
            parts = (Path(d) / "lockfile").read_text().strip().split(":")
            if len(parts) == 5 and parts[3]:
                return LcuAuth(int(parts[2]), parts[3])
        except (OSError, ValueError):
            continue
    return None


def _from_process() -> LcuAuth | None:
    for proc in psutil.process_iter(["name", "cmdline"]):
        if proc.info["name"] not in ("LeagueClientUx.exe", "LeagueClientUx"):
            continue
        cmd = " ".join(proc.info["cmdline"] or [])
        port = re.search(r"--app-port=(\d+)", cmd)
        token = re.search(r"--remoting-auth-token=([\w-]+)", cmd)
        if port and token:
            return LcuAuth(int(port.group(1)), token.group(1))
    return None


class Lcu:
    def __init__(self) -> None:
        self._auth: LcuAuth | None = None
        self._next_discovery = 0.0
        self._http = requests.Session()
        self._http.verify = False

    def _authenticate(self) -> bool:
        if self._auth is not None:
            return True
        if time.monotonic() < self._next_discovery:
            return False
        self._auth = _from_lockfile() or _from_process()
        if self._auth is None:
            self._next_discovery = time.monotonic() + _DISCOVERY_COOLDOWN
            return False
        self._http.auth = ("riot", self._auth.password)
        return True

    def get(self, path: str):
        """GET an LCU endpoint. Returns parsed JSON, or None if unavailable."""
        for _ in range(2):  # retry once with fresh credentials after a client restart
            if not self._authenticate():
                return None
            try:
                r = self._http.get(f"https://127.0.0.1:{self._auth.port}{path}", timeout=3)
            except requests.RequestException:
                self._auth = None
                continue
            if r.status_code == 401:
                self._auth = None
                continue
            try:
                return r.json() if r.ok else None
            except ValueError:
                return None
        return None

    def phase(self) -> str | None:
        """e.g. "None", "Lobby", "ChampSelect", "InProgress"."""
        return self.get("/lol-gameflow/v1/gameflow-phase")

    def queue_id(self) -> int | None:
        s = self.get("/lol-gameflow/v1/session") or {}
        return (s.get("gameData") or {}).get("queue", {}).get("id")
