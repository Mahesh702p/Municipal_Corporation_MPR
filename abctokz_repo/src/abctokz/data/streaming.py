# Augenblick — abctokz
"""Streaming corpus utilities for large datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def stream_shards(directory: str | Path, pattern: str = "*.txt") -> Iterator[str]:
    """Stream lines from all matching shards in *directory*.

    Args:
        directory: Directory containing corpus shard files.
        pattern: Glob pattern for shard files (default ``"*.txt"``).

    Yields:
        Lines from all shards in sorted order.
    """
    shards = sorted(Path(directory).glob(pattern))
    for shard in shards:
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


def batched(iterable: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    """Yield lists of *batch_size* items from *iterable*.

    The last batch may be smaller than *batch_size*.

    Args:
        iterable: Source iterator.
        batch_size: Number of items per batch.

    Yields:
        Lists of strings.
    """
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
