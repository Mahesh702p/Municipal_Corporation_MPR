# Augenblick — abctokz
"""Unicode helper utilities, with Devanagari-focused helpers."""

from __future__ import annotations

import unicodedata

from abctokz.constants import (
    DEVANAGARI_END,
    DEVANAGARI_EXTENDED_END,
    DEVANAGARI_EXTENDED_START,
    DEVANAGARI_START,
    VEDIC_EXTENSIONS_END,
    VEDIC_EXTENSIONS_START,
)


def is_devanagari(char: str) -> bool:
    """Return True if *char* belongs to a Devanagari Unicode block.

    Covers the core Devanagari block (U+0900–U+097F), Devanagari Extended
    (U+A8E0–U+A8FF), and Vedic Extensions (U+1CD0–U+1CFF).

    Args:
        char: Single Unicode character.

    Returns:
        ``True`` if the character is Devanagari.
    """
    cp = ord(char)
    return (
        (DEVANAGARI_START <= cp <= DEVANAGARI_END)
        or (DEVANAGARI_EXTENDED_START <= cp <= DEVANAGARI_EXTENDED_END)
        or (VEDIC_EXTENSIONS_START <= cp <= VEDIC_EXTENSIONS_END)
    )


def is_combining(char: str) -> bool:
    """Return True if *char* is a Unicode combining character.

    Args:
        char: Single Unicode character.

    Returns:
        ``True`` if the character category starts with 'M' (Mark).
    """
    return unicodedata.category(char).startswith("M")


def is_zero_width(char: str) -> bool:
    """Return True if *char* is a zero-width Unicode character.

    Covers zero-width non-joiner (U+200C), zero-width joiner (U+200D),
    and zero-width no-break space / BOM (U+FEFF).

    Args:
        char: Single Unicode character.

    Returns:
        ``True`` if the character is zero-width.
    """
    return ord(char) in {0x200C, 0x200D, 0xFEFF}


def grapheme_clusters(text: str) -> list[str]:
    """Split *text* into a list of Unicode grapheme clusters.

    This is a simplified implementation that groups a base character with
    any immediately following combining marks (category M). For production
    use, ``grapheme`` or ``regex`` with ``\\X`` is recommended.

    Args:
        text: Input Unicode string.

    Returns:
        List of grapheme clusters (each is one or more characters).
    """
    clusters: list[str] = []
    buf = ""
    for char in text:
        # IMPROVEMENT: Also attach zero-width characters (ZWJ/ZWNJ) to the cluster
        if (is_combining(char) or is_zero_width(char)) and buf:
            buf += char
        else:
            if buf:
                clusters.append(buf)
            buf = char
    if buf:
        clusters.append(buf)
    return clusters


def normalize_nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization.

    Args:
        text: Input string.

    Returns:
        NFKC-normalized string.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_nfc(text: str) -> str:
    """Apply Unicode NFC normalization.

    Args:
        text: Input string.

    Returns:
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


def strip_zero_width(text: str) -> str:
    """Remove all zero-width characters from *text*.

    Args:
        text: Input string.

    Returns:
        String with zero-width characters removed.
    """
    return "".join(c for c in text if not is_zero_width(c))
