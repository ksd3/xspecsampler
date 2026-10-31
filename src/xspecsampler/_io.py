"""File and hashing utilities used by the package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(value) -> str:
    """Return the SHA-256 hash of the canonical JSON form of a value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, value):
    """Write JSON to a temporary file and rename it, so that the file is never seen partly written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
