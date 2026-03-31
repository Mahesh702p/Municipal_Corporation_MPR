# Augenblick — abctokz
"""Content-hashing utilities for artifact integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Compute a deterministic SHA-256 digest of a JSON-serializable object.

    The object is serialized with sorted keys and no extra whitespace to
    ensure identical hashes for semantically equal objects.

    Args:
        obj: JSON-serializable Python object.

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
