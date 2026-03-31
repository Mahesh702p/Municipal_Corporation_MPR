# Augenblick — abctokz
"""Vocabulary management: token ↔ ID bijection."""

from __future__ import annotations

from abctokz.constants import UNK_TOKEN
from abctokz.exceptions import UnknownTokenError, VocabError
from abctokz.types import InverseVocabType, TokenID, VocabType


class Vocabulary:
    """Bidirectional token ↔ ID mapping.

    Maintains a deterministic bijection between string tokens and integer IDs.
    Special tokens are stored separately and always take priority.

    Args:
        vocab: Mapping from token string to integer ID.
        unk_token: The string used as the unknown token. If ``None``, looking
            up an OOV token raises :class:`~abctokz.exceptions.UnknownTokenError`.

    Example::

        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        assert vocab.token_to_id("hello") == 1
        assert vocab.id_to_token(2) == "world"
        assert vocab.token_to_id("foo") == 0  # falls back to <unk>
    """

    def __init__(self, vocab: VocabType, unk_token: str | None = UNK_TOKEN) -> None:
        self._vocab: VocabType = dict(vocab)
        self._inv: InverseVocabType = {v: k for k, v in self._vocab.items()}
        if len(self._vocab) != len(self._inv):
            raise VocabError("Vocabulary contains duplicate IDs.")
        self._unk_token = unk_token
        self._unk_id: TokenID | None = (
            self._vocab.get(unk_token) if unk_token is not None else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def unk_token(self) -> str | None:
        """The unknown token string, or ``None`` if not set."""
        return self._unk_token

    @property
    def unk_id(self) -> TokenID | None:
        """The unknown token ID, or ``None`` if not set."""
        return self._unk_id

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> TokenID:
        """Look up the ID for *token*.

        Args:
            token: Token string to look up.

        Returns:
            Corresponding integer ID.

        Raises:
            :class:`~abctokz.exceptions.UnknownTokenError`: If *token* is not in
                the vocabulary and no ``unk_token`` is configured.
        """
        result = self._vocab.get(token)
        if result is not None:
            return result
        if self._unk_id is not None:
            return self._unk_id
        raise UnknownTokenError(token)

    def id_to_token(self, token_id: TokenID) -> str:
        """Look up the string for *token_id*.

        Args:
            token_id: Integer ID to look up.

        Returns:
            Corresponding token string.

        Raises:
            :class:`~abctokz.exceptions.VocabError`: If *token_id* is not in the
                vocabulary.
        """
        token = self._inv.get(token_id)
        if token is None:
            raise VocabError(f"ID {token_id} not in vocabulary.")
        return token

    def __contains__(self, token: str) -> bool:
        return token in self._vocab

    def __len__(self) -> int:
        return len(self._vocab)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> VocabType:
        """Return a plain dict copy of the vocabulary."""
        return dict(self._vocab)

    @classmethod
    def from_dict(cls, data: VocabType, unk_token: str | None = UNK_TOKEN) -> "Vocabulary":
        """Construct a :class:`Vocabulary` from a plain dict.

        Args:
            data: Token-to-ID mapping.
            unk_token: Unknown token string.

        Returns:
            New :class:`Vocabulary` instance.
        """
        return cls(data, unk_token=unk_token)
