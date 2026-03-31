# Augenblick — abctokz
"""Devanagari-aware pre-tokenizer.

Splits on whitespace and, optionally, on script-change boundaries
(e.g. between Latin and Devanagari characters in the same word).
Grapheme cluster integrity is preserved: combining marks (matras, halant,
anusvara, etc.) stay attached to their base characters.
"""

from __future__ import annotations

from abctokz.pretokenizers.base import PreTokenizer
from abctokz.utils.unicode import grapheme_clusters, is_devanagari


def _script_of(cluster: str) -> str:
    """Return a coarse script label for a grapheme cluster.

    Returns ``"devanagari"``, ``"latin"``, or ``"other"``.

    Args:
        cluster: A grapheme cluster (one or more characters).

    Returns:
        Script label string.
    """
    base = cluster[0]  # classify by the base character
    if is_devanagari(base):
        return "devanagari"
    if base.isalpha():
        return "latin"
    return "other"


class DevanagariAwarePreTokenizer(PreTokenizer):
    """Pre-tokenizer that respects Devanagari grapheme boundaries.

    Splits on whitespace (always) and optionally on transitions between
    scripts (Devanagari ↔ Latin ↔ other).  Within a run of the same script,
    grapheme clusters are kept together, preserving the integrity of matras
    and other combining marks.

    Args:
        split_on_whitespace: Split on whitespace boundaries. Defaults to ``True``.
        split_on_script_boundary: Insert a split when the script changes within
            a whitespace-delimited token. Defaults to ``True``.

    Example::

        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("नमस्ते world")
        assert result == ["नमस्ते", "world"]

        result2 = pt.pre_tokenize("नमस्तेworld")
        assert result2 == ["नमस्ते", "world"]
    """

    def __init__(
        self,
        split_on_whitespace: bool = True,
        split_on_script_boundary: bool = True,
    ) -> None:
        self._split_ws = split_on_whitespace
        self._split_script = split_on_script_boundary

    def pre_tokenize(self, text: str) -> list[str]:
        """Split *text* into script-aware pre-tokens.

        Args:
            text: Normalized input string.

        Returns:
            List of pre-token strings.
        """
        # First pass: split on whitespace
        if self._split_ws:
            words = text.split()
        else:
            words = [text]

        if not self._split_script:
            return [w for w in words if w]

        # Second pass: split each word on script boundaries
        result: list[str] = []
        for word in words:
            result.extend(self._split_by_script(word))
        return [t for t in result if t]

    def _split_by_script(self, word: str) -> list[str]:
        """Split a single word on script-change boundaries.

        Grapheme cluster integrity is preserved.

        Args:
            word: Single whitespace-delimited token.

        Returns:
            List of script-homogeneous substrings.
        """
        clusters = grapheme_clusters(word)
        if not clusters:
            return []

        parts: list[str] = []
        buf = clusters[0]
        prev_script = _script_of(clusters[0])

        for cluster in clusters[1:]:
            script = _script_of(cluster)
            # Only split when the script actually changes between named scripts
            # (devanagari <-> latin). "other" clusters (digits, punctuation)
            # are attached to the current script group.
            if script != "other" and prev_script != "other" and script != prev_script:
                parts.append(buf)
                buf = cluster
                prev_script = script
            else:
                buf += cluster
                if script != "other":
                    prev_script = script

        if buf:
            parts.append(buf)
        return parts
