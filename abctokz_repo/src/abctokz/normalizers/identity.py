# Augenblick — abctokz
"""Identity (pass-through) normalizer."""

from __future__ import annotations

from abctokz.normalizers.base import Normalizer


class IdentityNormalizer(Normalizer):
    """Returns the input string unchanged.

    Useful as a no-op placeholder or when normalization must be completely
    disabled for byte-level or raw-text tokenizers.

    Example::

        norm = IdentityNormalizer()
        assert norm.normalize("Hello!") == "Hello!"
    """

    def normalize(self, text: str) -> str:
        """Return *text* unchanged.

        Args:
            text: Any string.

        Returns:
            The same string, unmodified.
        """
        return text
