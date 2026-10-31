"""Context-induced explanation instability in tabular foundation models."""

import os
from pathlib import Path


def load_env(path: Path | None = None) -> None:
    """Copy repo-local .env entries into os.environ (existing values win)."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# Must run before tabpfn / tabicl / openml read their cache settings.
load_env()
