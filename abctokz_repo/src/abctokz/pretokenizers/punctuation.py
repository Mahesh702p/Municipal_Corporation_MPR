# Augenblick — abctokz
"""Punctuation-aware pre-tokenizer."""

from __future__ import annotations

import re
import unicodedata

from abctokz.pretokenizers.base import PreTokenizer

# Matches a run of Unicode punctuation characters
_PUNCT_RE = re.compile(r"\p{P}+", re.UNICODE) if False else None  # placeholder

try:
    import regex as _regex  # type: ignore[import-untyped]

    _PUNCT_SPLITTER = _regex.compile(r"(\p{P})")
except ImportError:
    # Fallback: ASCII punctuation only
    _PUNCT_SPLITTER = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")


def _is_punctuation(char: str) -> bool:
    """Return True if *char* is a Unicode punctuation character."""
    cat = unicodedata.category(char)
    # P* categories: Pc, Pd, Pe, Pf, Pi, Po, Ps
    return cat.startswith("P") or char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


class PunctuationPreTokenizer(PreTokenizer):
    """Split text at punctuation boundaries.

    Args:
        behavior: How to handle punctuation characters:
            - ``"isolated"``: punctuation becomes its own token.
            - ``"merged_with_previous"``: punctuation is attached to the
              preceding word token.
            - ``"merged_with_next"``: punctuation is attached to the
              following word token.

    Example::

        pt = PunctuationPreTokenizer(behavior="isolated")
        assert pt.pre_tokenize("hello, world!") == ["hello", ",", "world", "!"]
    """

    def __init__(
        self,
        behavior: str = "isolated",
    ) -> None:
        if behavior not in ("isolated", "merged_with_previous", "merged_with_next"):
            raise ValueError(f"Unknown punctuation behavior: {behavior!r}")
        self._behavior = behavior

    def pre_tokenize(self, text: str) -> list[str]:
        """Split *text* at punctuation boundaries.

        Args:
            text: Normalized input string.

        Returns:
            List of token strings.
        """
        # Split on whitespace first, then handle punctuation per word
        words = text.split()
        result: list[str] = []
        for word in words:
            result.extend(self._split_word(word))
        return [t for t in result if t]

    def _split_word(self, word: str) -> list[str]:
        """Split a single whitespace-delimited word on punctuation."""
        parts = _PUNCT_SPLITTER.split(word)
        parts = [p for p in parts if p]

        if self._behavior == "isolated":
            return parts

        merged: list[str] = []
        if self._behavior == "merged_with_previous":
            buf = ""
            for p in parts:
                if _is_punctuation(p[0]) and buf:
                    buf += p
                else:
                    if buf:
                        merged.append(buf)
                    buf = p
            if buf:
                merged.append(buf)
            return merged

        # merged_with_next
        buf = ""
        for p in reversed(parts):
            if _is_punctuation(p[0]):
                buf = p + buf
            else:
                merged.insert(0, p + buf)
                buf = ""
        if buf:
            merged.insert(0, buf)
        return merged
