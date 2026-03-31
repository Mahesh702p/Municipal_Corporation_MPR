# Augenblick — abctokz
"""Abstract base class for all abctokz tokenization models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Model(ABC):
    """Abstract base for all tokenization models.

    A model is responsible for tokenizing a single pre-token (a whitespace-
    delimited word or script segment) into a list of ``(token, id)`` pairs.
    The model is stateless with respect to the full input sequence—it only
    sees one pre-token at a time.

    Subclasses implement :meth:`tokenize`, :meth:`save`, and :meth:`load`.

    Example::

        class MyModel(Model):
            def tokenize(self, sequence: str) -> list[tuple[str, int]]:
                ...
            def save(self, directory: str | Path) -> None:
                ...
            @classmethod
            def load(cls, directory: str | Path) -> "MyModel":
                ...
    """

    @abstractmethod
    def tokenize(self, sequence: str) -> list[tuple[str, int]]:
        """Tokenize a single *sequence* (pre-token) into ``(token, id)`` pairs.

        Args:
            sequence: A single pre-token string (already pre-tokenized).

        Returns:
            Ordered list of ``(token_string, token_id)`` tuples.
        """

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Persist model artifacts to *directory*.

        Args:
            directory: Target directory path.
        """

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "Model":
        """Load model artifacts from *directory*.

        Args:
            directory: Source directory path.

        Returns:
            Instantiated model.
        """

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-to-ID vocabulary as a plain dict.

        Returns:
            Vocabulary mapping.
        """
        return {}

    def get_vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary.

        Returns:
            Vocabulary size.
        """
        return len(self.get_vocab())
