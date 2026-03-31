# Augenblick — abctokz
"""Sequence (pipeline) normalizer that chains multiple normalizers."""

from __future__ import annotations

from abctokz.normalizers.base import Normalizer


class SequenceNormalizer(Normalizer):
    """Apply a sequence of normalizers in order.

    Each normalizer receives the output of the previous one.  An empty
    sequence acts as an identity normalizer.

    Args:
        normalizers: Ordered list of :class:`~abctokz.normalizers.base.Normalizer`
            instances to apply in sequence.

    Example::

        from abctokz.normalizers import NfkcNormalizer, WhitespaceNormalizer
        norm = SequenceNormalizer([NfkcNormalizer(), WhitespaceNormalizer()])
        result = norm.normalize("  ＨＥＬＬＯ  ")
        assert result == "HELLO"
    """

    def __init__(self, normalizers: list[Normalizer]) -> None:
        self._normalizers = list(normalizers)

    @property
    def normalizers(self) -> list[Normalizer]:
        """The ordered list of child normalizers."""
        return self._normalizers

    def normalize(self, text: str) -> str:
        """Apply each normalizer in sequence.

        Args:
            text: Input string.

        Returns:
            String after all normalizers have been applied.
        """
        for norm in self._normalizers:
            text = norm.normalize(text)
        return text
