# Augenblick — abctokz
"""Lightweight timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timed(label: str = "") -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed wall-clock time.

    Args:
        label: Optional descriptive label (unused internally, for caller context).

    Yields:
        A mutable dict with key ``"elapsed"`` that is populated on exit.

    Example::

        with timed("encoding") as t:
            result = tokenizer.encode(text)
        print(t["elapsed"])  # seconds
    """
    info: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed"] = time.perf_counter() - start


def throughput(n_items: int, elapsed_seconds: float) -> float:
    """Compute items-per-second throughput.

    Args:
        n_items: Number of items processed.
        elapsed_seconds: Time taken in seconds.

    Returns:
        Items per second, or 0.0 if elapsed is non-positive.
    """
    if elapsed_seconds <= 0:
        return 0.0
    return n_items / elapsed_seconds
