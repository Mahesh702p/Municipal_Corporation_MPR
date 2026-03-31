# Augenblick — abctokz
"""Regex-based pre-tokenizer."""

from __future__ import annotations

import re
from typing import Pattern

from abctokz.pretokenizers.base import PreTokenizer


class RegexPreTokenizer(PreTokenizer):
    """Split (or keep) text using a configurable regex pattern.

    Args:
        pattern: Regular expression string.
        invert: If ``False`` (default), split on matches; resulting tokens are
            the *gaps* between matches.  If ``True``, keep matches as tokens
            (findall behaviour).

    Example::

        # Split on whitespace or punctuation
        pt = RegexPreTokenizer(r"[\\s]+")
        assert pt.pre_tokenize("foo  bar") == ["foo", "bar"]

        # Keep word-like sequences
        pt2 = RegexPreTokenizer(r"[\\w]+", invert=True)
        assert pt2.pre_tokenize("hello, world!") == ["hello", "world"]
    """

    def __init__(self, pattern: str, invert: bool = False) -> None:
        try:
            import regex as _re  # type: ignore[import-untyped]
            self._re: Pattern[str] = _re.compile(pattern)
        except ImportError:
            self._re = re.compile(pattern)
        self._invert = invert

    def pre_tokenize(self, text: str) -> list[str]:
        """Split or find tokens in *text* using the configured pattern.

        Args:
            text: Normalized input string.

        Returns:
            List of token strings.
        """
        if self._invert:
            return self._re.findall(text)
        tokens = self._re.split(text)
        return [t for t in tokens if t]
