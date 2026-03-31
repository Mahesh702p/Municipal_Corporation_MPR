# Augenblick — abctokz
"""Whitespace normalization."""

from __future__ import annotations

import re

from abctokz.normalizers.base import Normalizer

_MULTI_WS = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\n{2,}")


class WhitespaceNormalizer(Normalizer):
    """Normalizes whitespace in input text.

    Optionally strips leading/trailing whitespace and collapses runs of
    spaces/tabs into a single space.  Newlines are preserved by default.

    Args:
        strip: Strip leading and trailing whitespace.
        collapse: Collapse multiple consecutive spaces/tabs into one.

    Example::

        norm = WhitespaceNormalizer()
        assert norm.normalize("  hello   world  ") == "hello world"
    """

    def __init__(self, strip: bool = True, collapse: bool = True) -> None:
        self._strip = strip
        self._collapse = collapse

    def normalize(self, text: str) -> str:
        """Normalize whitespace in *text*.

        Args:
            text: Input string.

        Returns:
            Whitespace-normalized string.
        """
        if self._collapse:
            text = _MULTI_WS.sub(" ", text)
        if self._strip:
            text = text.strip()
        return text
