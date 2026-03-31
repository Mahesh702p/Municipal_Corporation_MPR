# Augenblick — abctokz
"""Devanagari-safe normalizer.

Design notes
------------
*  We apply NFC (not NFKC) because NFKC can collapse Devanagari combining
   marks in ways that change the visual and phonetic form of the character.
*  Zero-width characters (ZWJ U+200D, ZWNJ U+200C) are *preserved by default*
   because they are phonetically and visually meaningful in Devanagari (they
   control conjunct formation). Set ``strip_zero_width=True`` to remove them.
*  The normalizer is **lossless relative to the normalized form**: encoding
   and decoding will faithfully reproduce the NFC-normalized input. If you
   need round-trip losslessness to the raw input, use
   :class:`~abctokz.normalizers.identity.IdentityNormalizer`.
*  Chandrabindu (U+0900), Anusvara (U+0902), Visarga (U+0903), and Nukta
   (U+093C) are combining marks and are kept attached to their base characters
   by the grapheme-cluster-aware logic in
   :mod:`abctokz.utils.unicode`.
"""

from __future__ import annotations

import unicodedata

from abctokz.normalizers.base import Normalizer
from abctokz.utils.unicode import normalize_nfc, strip_zero_width


class DevanagariNormalizer(Normalizer):
    """NFC-based normalizer safe for Devanagari script.

    Args:
        nfc_first: Apply NFC normalization before Devanagari rules.
            Defaults to ``True``.
        strip_zero_width: Remove ZWJ (U+200D) and ZWNJ (U+200C).
            Defaults to ``False`` to preserve conjunct-control semantics.

    Example::

        norm = DevanagariNormalizer()
        text = "नमस्ते"
        assert norm.normalize(text) == unicodedata.normalize("NFC", text)
    """

    def __init__(self, nfc_first: bool = True, strip_zero_width: bool = False) -> None:
        self._nfc_first = nfc_first
        self._strip_zw = strip_zero_width

    def normalize(self, text: str) -> str:
        """Normalize *text* with Devanagari-safe rules.

        Processing order:

        1. (Optional) NFC normalization.
        2. (Optional) Strip zero-width characters.
        3. Normalize whitespace sequences that are non-Devanagari spaces
           (e.g. ideographic space U+3000) to ASCII space.

        Args:
            text: Input string, may contain Devanagari and/or Latin characters.

        Returns:
            Normalized string.
        """
        if self._nfc_first:
            text = normalize_nfc(text)
        if self._strip_zw:
            text = strip_zero_width(text)
        # Normalize ideographic/exotic whitespace to regular space
        text = _normalize_exotic_whitespace(text)
        return text


# Characters that should be treated as regular spaces
_EXOTIC_SPACES = {
    "\u00A0",  # NO-BREAK SPACE
    "\u1680",  # OGHAM SPACE MARK
    "\u2000",  # EN QUAD
    "\u2001",  # EM QUAD
    "\u2002",  # EN SPACE
    "\u2003",  # EM SPACE
    "\u2004",  # THREE-PER-EM SPACE
    "\u2005",  # FOUR-PER-EM SPACE
    "\u2006",  # SIX-PER-EM SPACE
    "\u2007",  # FIGURE SPACE
    "\u2008",  # PUNCTUATION SPACE
    "\u2009",  # THIN SPACE
    "\u200A",  # HAIR SPACE
    "\u202F",  # NARROW NO-BREAK SPACE
    "\u205F",  # MEDIUM MATHEMATICAL SPACE
    "\u3000",  # IDEOGRAPHIC SPACE
}


def _normalize_exotic_whitespace(text: str) -> str:
    """Replace exotic Unicode space characters with ASCII space."""
    return "".join(" " if c in _EXOTIC_SPACES else c for c in text)
