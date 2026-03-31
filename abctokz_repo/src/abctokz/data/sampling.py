# Augenblick — abctokz
"""Corpus sampling utilities."""

from __future__ import annotations

import random
from typing import Sequence


def sample_lines(
    lines: Sequence[str],
    n: int,
    seed: int = 42,
) -> list[str]:
    """Sample *n* lines from *lines* without replacement.

    If ``n >= len(lines)``, returns a shuffled copy of all lines.

    Args:
        lines: Source lines.
        n: Number of lines to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled lines.
    """
    rng = random.Random(seed)
    population = list(lines)
    if n >= len(population):
        rng.shuffle(population)
        return population
    return rng.sample(population, n)


def stratified_sample(
    lines_by_lang: dict[str, list[str]],
    n_per_lang: int,
    seed: int = 42,
) -> list[str]:
    """Sample up to *n_per_lang* lines from each language bucket.

    Args:
        lines_by_lang: Mapping from language tag to lines.
        n_per_lang: Maximum lines per language.
        seed: Random seed.

    Returns:
        Combined list of sampled lines (interleaved by language).
    """
    result: list[str] = []
    for lang, lines in sorted(lines_by_lang.items()):
        result.extend(sample_lines(lines, n_per_lang, seed=seed))
    return result
