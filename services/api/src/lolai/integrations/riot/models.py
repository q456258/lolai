from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChampSelect:
    """Snapshot of the champion-select lobby, as seen by the local player."""

    my_champion: str | None
    my_position: str = ""                                   # TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY, "" in ARAM
    allies: list[str] = field(default_factory=list)         # other allies' picks/hovers
    enemies: list[str] = field(default_factory=list)        # hidden in ARAM, filled in draft modes
    bench: list[str] = field(default_factory=list)          # ARAM bench you can swap with
    bans: list[str] = field(default_factory=list)
    pickable: list[str] = field(default_factory=list)       # draft: champions you may still pick
    rerolls_left: int = 0
    my_action: str | None = None                            # "pick" / "ban" while it is my turn

    def key(self) -> tuple:
        """Changes whenever anything worth re-analysing changes."""
        return (
            self.my_champion,
            self.my_action,
            self.rerolls_left,
            tuple(sorted(self.allies)),
            tuple(sorted(self.enemies)),
            tuple(sorted(self.bench)),
            tuple(sorted(self.bans)),
        )


@dataclass(frozen=True)
class LiveGame:
    """Snapshot of the in-game state from the Live Client Data API."""

    my_champion: str
    level: int
    allies: list[str]
    enemies: list[str]
    my_items: list[str]
