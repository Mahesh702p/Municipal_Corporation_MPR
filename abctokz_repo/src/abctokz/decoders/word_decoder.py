# Augenblick — abctokz
"""Word-level decoder: joins tokens with spaces."""

from __future__ import annotations

from abctokz.decoders.base import Decoder


class WordDecoder(Decoder):
    """Decodes a sequence of word tokens by joining with spaces.

    This decoder is appropriate for :class:`~abctokz.models.wordlevel.WordLevelModel`
    outputs. Special tokens (like ``<unk>``, ``<s>``, ``</s>``) can optionally
    be excluded.

    Args:
        separator: String inserted between tokens. Defaults to ``" "``.
        skip_special_tokens: If ``True``, tokens that start and end with ``<>``
            are excluded from the output. Defaults to ``False``.

    Example::

        dec = WordDecoder()
        assert dec.decode(["hello", "world"]) == "hello world"
    """

    def __init__(self, separator: str = " ", skip_special_tokens: bool = False) -> None:
        self._sep = separator
        self._skip_special = skip_special_tokens

    def decode(self, tokens: list[str]) -> str:
        """Join *tokens* with the configured separator.

        Args:
            tokens: Token strings to join.

        Returns:
            Decoded string.
        """
        if self._skip_special:
            tokens = [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]
        return self._sep.join(tokens)
