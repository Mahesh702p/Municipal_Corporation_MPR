# Augenblick — abctokz
"""Determinism and seed utilities for abctokz."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducible training.

    Currently sets the Python :mod:`random` module seed.
    Extend here if NumPy or other sources of randomness are added.

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    try:
        import numpy as np  # type: ignore[import-untyped]

        np.random.seed(seed)
    except ImportError:
        pass
