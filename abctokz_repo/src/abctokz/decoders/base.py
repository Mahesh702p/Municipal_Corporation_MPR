# Augenblick — abctokz
"""Abstract base class for all abctokz decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Decoder(ABC):
    """Abstract base for all decoders.

    A decoder reconstructs a string from a list of token strings.
    Decoders are stateless and deterministic.
    """

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """Decode *tokens* into a string.

        Args:
            tokens: Ordered list of token strings.

        Returns:
            Reconstructed string.
        """

    def __call__(self, tokens: list[str]) -> str:
        """Alias for :meth:`decode`."""
        return self.decode(tokens)
