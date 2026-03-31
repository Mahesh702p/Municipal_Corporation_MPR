# Augenblick — abctokz
"""Unicode NFKC normalizer."""

from __future__ import annotations

from abctokz.normalizers.base import Normalizer
from abctokz.utils.unicode import normalize_nfkc, strip_zero_width


class NfkcNormalizer(Normalizer):
    """Applies Unicode NFKC normalization.

    NFKC (Compatibility Decomposition followed by Canonical Composition)
    folds compatibility characters (e.g. fullwidth ASCII, ligatures) into
    their canonical equivalents. This is the default normalization used by
    many multilingual tokenizers.

    .. warning::
        NFKC can be lossy for some Devanagari text.  For Devanagari input,
        prefer :class:`~abctokz.normalizers.devanagari.DevanagariNormalizer`
        which uses NFC instead.

    Args:
        strip_zero_width: If ``True`` (default), remove zero-width characters
            (ZWJ, ZWNJ, BOM) after NFKC normalization.

    Example::

        norm = NfkcNormalizer()
        assert norm.normalize("ＨＥＬＬＯ") == "HELLO"
    """

    def __init__(self, strip_zero_width: bool = True) -> None:
        self._strip_zw = strip_zero_width

    def normalize(self, text: str) -> str:
        """Apply NFKC normalization, optionally stripping zero-width chars.

        Args:
            text: Input string.

        Returns:
            Normalized string.
        """
        result = normalize_nfkc(text)
        if self._strip_zw:
            result = strip_zero_width(result)
        return result
