"""Single place that decides where the stats database lives (independent of module nesting)."""

from pathlib import Path

# riot/ -> integrations/ -> lolai/ -> src/ -> api/ -> services/ -> repo root
DB_PATH = Path(__file__).resolve().parents[6] / "data" / "stats.sqlite"
