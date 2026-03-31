# Augenblick — abctokz
"""Subword decoder for BPE and Unigram models."""

from __future__ import annotations

from abctokz.constants import BPE_CONTINUATION_PREFIX
from abctokz.decoders.base import Decoder


class SubwordDecoder(Decoder):
    """Decodes subword token sequences into strings.

    Handles both BPE-style continuation prefixes (``##``) and
    SentencePiece-style word-start prefixes (``▁``).

    **BPE (continuation prefix) mode**: Non-initial pieces are prefixed with
    ``continuation_prefix`` (default ``"##"``). The decoder strips these
    prefixes and concatenates adjacent pieces; word boundaries are inferred
    from the absence of the prefix.

    **SentencePiece/Unigram mode**: Word-initial pieces are prefixed with
    ``"▁"`` (U+2581). The decoder replaces ``▁`` with a space (except at
    the very beginning).

    Args:
        continuation_prefix: Prefix used for non-initial BPE pieces.
        space_prefix: Prefix used for word-initial pieces (SentencePiece style).
        skip_special_tokens: Skip tokens that look like ``<special>`` tokens.

    Example::

        dec = SubwordDecoder(continuation_prefix="##")
        assert dec.decode(["hello", "##world"]) == "helloworld"
        assert dec.decode(["good", "bye"]) == "good bye"

        dec2 = SubwordDecoder(space_prefix="▁")
        assert dec2.decode(["▁hello", "▁world"]) == "hello world"
    """

    def __init__(
        self,
        continuation_prefix: str = BPE_CONTINUATION_PREFIX,
        space_prefix: str = "",
        skip_special_tokens: bool = False,
    ) -> None:
        self._cont_prefix = continuation_prefix
        self._space_prefix = space_prefix
        self._skip_special = skip_special_tokens

    def decode(self, tokens: list[str]) -> str:
        """Reconstruct text from subword *tokens*.

        Args:
            tokens: Ordered subword token strings.

        Returns:
            Decoded text string.
        """
        if self._skip_special:
            tokens = [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]

        if not tokens:
            return ""

        # SentencePiece / space-prefix mode
        if self._space_prefix:
            parts: list[str] = []
            for i, tok in enumerate(tokens):
                if tok.startswith(self._space_prefix):
                    stripped = tok[len(self._space_prefix):]
                    if i == 0:
                        parts.append(stripped)
                    else:
                        parts.append(" " + stripped)
                else:
                    parts.append(tok)
            return "".join(parts)

        # BPE continuation-prefix mode
        parts = []
        for tok in tokens:
            if tok.startswith(self._cont_prefix) and self._cont_prefix:
                parts.append(tok[len(self._cont_prefix):])
            else:
                if parts:
                    parts.append(" ")
                parts.append(tok)
        return "".join(parts)
