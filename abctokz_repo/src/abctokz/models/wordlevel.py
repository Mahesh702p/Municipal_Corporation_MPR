# Augenblick — abctokz
"""WordLevel tokenization model."""

from __future__ import annotations

from pathlib import Path

from abctokz.constants import UNK_TOKEN
from abctokz.models.base import Model
from abctokz.vocab.serialization import load_vocab, save_vocab
from abctokz.vocab.vocab import Vocabulary


class WordLevelModel(Model):
    """Simple word-level tokenizer model.

    Each pre-token is looked up directly in the vocabulary. Unknown tokens
    map to the ``<unk>`` ID. This model does *not* perform subword
    segmentation—it simply maps whole words to IDs.

    Args:
        vocab: :class:`~abctokz.vocab.vocab.Vocabulary` mapping tokens to IDs.
        unk_token: The unknown token string.

    Example::

        from abctokz.vocab import Vocabulary
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        model = WordLevelModel(vocab)
        assert model.tokenize("hello") == [("hello", 1)]
        assert model.tokenize("xyz") == [("<unk>", 0)]
    """

    def __init__(self, vocab: Vocabulary, unk_token: str = UNK_TOKEN) -> None:
        self._vocab = vocab
        self._unk_token = unk_token

    def tokenize(self, sequence: str) -> list[tuple[str, int]]:
        """Tokenize *sequence* by direct vocabulary lookup.

        Args:
            sequence: A single pre-token.

        Returns:
            Single-element list ``[(token, id)]``. If *sequence* is not in
            vocabulary, the ``<unk>`` token is returned.
        """
        token_id = self._vocab.token_to_id(sequence)
        if token_id == self._vocab.unk_id and sequence not in self._vocab:
            return [(self._unk_token, token_id)]
        return [(sequence, token_id)]

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary dict."""
        return self._vocab.to_dict()

    def save(self, directory: str | Path) -> None:
        """Save vocabulary to *directory*.

        Args:
            directory: Target directory.
        """
        save_vocab(self._vocab, directory)

    @classmethod
    def load(cls, directory: str | Path, unk_token: str = UNK_TOKEN) -> "WordLevelModel":
        """Load a :class:`WordLevelModel` from *directory*.

        Args:
            directory: Source directory containing ``vocab.json``.
            unk_token: Unknown token string.

        Returns:
            Loaded :class:`WordLevelModel`.
        """
        vocab = load_vocab(directory, unk_token=unk_token)
        return cls(vocab, unk_token=unk_token)
