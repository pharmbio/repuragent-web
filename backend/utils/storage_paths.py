"""Helpers for resolving persistent data directories."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    """
    Return the configured data directory.

    Falls back to the repository-local ``data`` folder if the ``DATA_ROOT``
    environment variable is not provided.
    """
    root = Path(os.environ.get("DATA_ROOT", "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
