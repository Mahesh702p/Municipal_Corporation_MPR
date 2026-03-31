# Augenblick — abctokz
"""Whitespace pre-tokenizer."""

from __future__ import annotations

from abctokz.pretokenizers.base import PreTokenizer


class WhitespacePreTokenizer(PreTokenizer):
    """Split text on any whitespace boundary.

    Uses Python's :meth:`str.split` (splits on runs of whitespace), so
    consecutive whitespace characters are treated as a single delimiter.
    Empty strings are never included in the output.

    Example::

        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("hello world") == ["hello", "world"]
        assert pt.pre_tokenize("  foo  bar  ") == ["foo", "bar"]
    """

    def pre_tokenize(self, text: str) -> list[str]:
        """Split *text* on whitespace.

        Args:
            text: Normalized input string.

        Returns:
            List of tokens, with whitespace removed.
        """
        return text.split()
