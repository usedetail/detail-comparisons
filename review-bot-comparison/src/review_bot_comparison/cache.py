"""Tiny JSON-backed cache. Keys are SHA-256 hashes of the prompt + input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")


def text_hash(text: str) -> str:
    """16-char SHA-256 prefix. Long enough to avoid collisions in this corpus."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class DiskCache:
    """JSON-backed dict-of-strings keyed by hash. Loaded lazily, saved on demand.

    Single-process; not safe for concurrent writers. Reads are thread-safe.
    """

    def __init__(self, name: str, directory: Path = DATA_DIR) -> None:
        self.path = directory / f"{name}.json"
        self.data: dict[str, Any] = {}

    def load(self) -> None:
        if self.path.exists():
            with self.path.open() as f:
                self.data = json.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()
