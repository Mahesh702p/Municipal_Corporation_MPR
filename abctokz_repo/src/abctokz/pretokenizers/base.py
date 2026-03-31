# Augenblick — abctokz
"""Abstract base class for all abctokz pre-tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PreTokenizer(ABC):
    """Abstract base for all pre-tokenizers.

    A pre-tokenizer splits the normalized string into a list of *pre-tokens*
    (raw strings), each of which the model will tokenize independently.
    Pre-tokenization defines the coarsest boundaries the model cannot cross.

    Pre-tokenizers are stateless and deterministic.

    Example::

        class MyPT(PreTokenizer):
            def pre_tokenize(self, text: str) -> list[str]:
                return text.split()
    """

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[str]:
        """Split *text* into a list of pre-token strings.

        Args:
            text: Normalized input string.

        Returns:
            Ordered list of pre-token strings.
        """

    def __call__(self, text: str) -> list[str]:
        """Alias for :meth:`pre_tokenize`."""
        return self.pre_tokenize(text)
