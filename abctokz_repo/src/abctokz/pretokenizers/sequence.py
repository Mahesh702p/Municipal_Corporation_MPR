# Augenblick — abctokz
"""Sequence (pipeline) pre-tokenizer that chains multiple pre-tokenizers."""

from __future__ import annotations

from abctokz.pretokenizers.base import PreTokenizer


class SequencePreTokenizer(PreTokenizer):
    """Apply a list of pre-tokenizers in sequence.

    Each pre-tokenizer receives the outputs of the previous one and may
    further split any of the tokens. The final result is the flattened list
    of all pre-tokens from the last stage.

    Args:
        pretokenizers: Ordered list of
            :class:`~abctokz.pretokenizers.base.PreTokenizer` instances.

    Example::

        from abctokz.pretokenizers import WhitespacePreTokenizer, PunctuationPreTokenizer
        pt = SequencePreTokenizer([
            WhitespacePreTokenizer(),
            PunctuationPreTokenizer(behavior="isolated"),
        ])
        result = pt.pre_tokenize("hello, world!")
        assert result == ["hello", ",", "world", "!"]
    """

    def __init__(self, pretokenizers: list[PreTokenizer]) -> None:
        self._pretokenizers = list(pretokenizers)

    @property
    def pretokenizers(self) -> list[PreTokenizer]:
        """The ordered list of child pre-tokenizers."""
        return self._pretokenizers

    def pre_tokenize(self, text: str) -> list[str]:
        """Apply each pre-tokenizer in sequence, flattening results.

        Args:
            text: Normalized input string.

        Returns:
            Flat list of final pre-tokens.
        """
        tokens: list[str] = [text]
        for pt in self._pretokenizers:
            new_tokens: list[str] = []
            for tok in tokens:
                new_tokens.extend(pt.pre_tokenize(tok))
            tokens = new_tokens
        return tokens
