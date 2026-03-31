# Augenblick — abctokz
"""Corpus loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_lines(path: str | Path, *, strip: bool = True, skip_empty: bool = True) -> Iterator[str]:
    """Iterate over lines in a text file.

    Args:
        path: Path to the text file.
        strip: Strip leading/trailing whitespace from each line.
        skip_empty: Skip empty lines after stripping.

    Yields:
        Individual lines from the file.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if strip:
                line = line.strip()
            if skip_empty and not line:
                continue
            yield line


def iter_corpus(
    paths: list[str | Path], *, strip: bool = True, skip_empty: bool = True
) -> Iterator[str]:
    """Iterate over lines across multiple corpus files.

    Args:
        paths: List of file paths.
        strip: Strip whitespace from each line.
        skip_empty: Skip empty lines.

    Yields:
        Lines from all files in order.
    """
    for path in paths:
        yield from iter_lines(path, strip=strip, skip_empty=skip_empty)


def load_corpus(paths: list[str | Path]) -> list[str]:
    """Load all lines from one or more files into a list.

    Args:
        paths: List of file paths.

    Returns:
        Flat list of lines.
    """
    return list(iter_corpus(paths))
