# Augenblick — abctokz
"""Abstract base class for all abctokz normalizers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Normalizer(ABC):
    """Abstract base for all normalizers.

    A normalizer transforms a raw input string into a canonical form before
    it is passed to the pre-tokenizer. Normalizers are stateless and
    deterministic—the same input always produces the same output.

    Example::

        class MyNorm(Normalizer):
            def normalize(self, text: str) -> str:
                return text.lower()
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize *text* and return the result.

        Args:
            text: Raw input string.

        Returns:
            Normalized string.
        """

    def __call__(self, text: str) -> str:
        """Alias for :meth:`normalize` to allow normalizer(text) syntax."""
        return self.normalize(text)
