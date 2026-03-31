# Augenblick — abctokz
"""File I/O utilities for loading and saving abctokz artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed Python object.

    Args:
        path: Path to the JSON file.

    Returns:
        Deserialized Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    """Serialize *obj* to a JSON file at *path*.

    Creates parent directories as needed.

    Args:
        obj: JSON-serializable Python object.
        path: Destination path.
        indent: Indentation level for pretty-printing. Defaults to 2.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


def load_text_lines(path: str | Path, *, strip: bool = True) -> list[str]:
    """Read all lines from a text file.

    Args:
        path: Path to the text file.
        strip: If ``True``, strip leading/trailing whitespace from each line
               and drop empty lines. Defaults to ``True``.

    Returns:
        List of lines.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    if strip:
        lines = [ln.strip() for ln in lines if ln.strip()]
    return lines


def ensure_dir(path: str | Path) -> Path:
    """Create *path* as a directory (including parents) if it does not exist.

    Args:
        path: Target directory path.

    Returns:
        The resolved :class:`~pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
