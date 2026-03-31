# Augenblick — abctokz
"""Abstract base class for post-processors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from abctokz.types import Encoding


class PostProcessor(ABC):
    """Abstract base for all post-processors.

    A post-processor takes an :class:`~abctokz.types.Encoding` (and optionally
    a pair encoding) and returns a new :class:`~abctokz.types.Encoding` with any
    desired modifications (e.g. adding special tokens, truncation, etc.).
    """

    @abstractmethod
    def process(self, encoding: Encoding, pair: Encoding | None = None) -> Encoding:
        """Apply post-processing to *encoding*.

        Args:
            encoding: The primary encoding.
            pair: Optional second encoding for sequence-pair tasks.

        Returns:
            Processed :class:`~abctokz.types.Encoding`.
        """

    def __call__(self, encoding: Encoding, pair: Encoding | None = None) -> Encoding:
        """Alias for :meth:`process`."""
        return self.process(encoding, pair)
