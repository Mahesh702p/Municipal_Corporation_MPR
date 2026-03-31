# Augenblick — abctokz
"""Normalizer subpackage for abctokz."""

from abctokz.normalizers.base import Normalizer
from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.identity import IdentityNormalizer
from abctokz.normalizers.sequence import SequenceNormalizer
from abctokz.normalizers.unicode_nfkc import NfkcNormalizer
from abctokz.normalizers.whitespace import WhitespaceNormalizer
from abctokz.config.schemas import (
    AnyNormalizerConfig,
    DevanagariNormalizerConfig,
    IdentityNormalizerConfig,
    NfkcNormalizerConfig,
    SequenceNormalizerConfig,
    WhitespaceNormalizerConfig,
)


def build_normalizer(config: AnyNormalizerConfig) -> Normalizer:
    """Construct a :class:`Normalizer` from a config object.

    Args:
        config: A validated normalizer config.

    Returns:
        Corresponding :class:`Normalizer` instance.

    Raises:
        ValueError: For unknown config types.
    """
    if isinstance(config, IdentityNormalizerConfig):
        return IdentityNormalizer()
    if isinstance(config, NfkcNormalizerConfig):
        return NfkcNormalizer(strip_zero_width=config.strip_zero_width)
    if isinstance(config, WhitespaceNormalizerConfig):
        return WhitespaceNormalizer(strip=config.strip, collapse=config.collapse)
    if isinstance(config, DevanagariNormalizerConfig):
        return DevanagariNormalizer(
            nfc_first=config.nfc_first, strip_zero_width=config.strip_zero_width
        )
    if isinstance(config, SequenceNormalizerConfig):
        return SequenceNormalizer([build_normalizer(c) for c in config.normalizers])
    raise ValueError(f"Unknown normalizer config type: {type(config)}")


__all__ = [
    "Normalizer",
    "DevanagariNormalizer",
    "IdentityNormalizer",
    "NfkcNormalizer",
    "SequenceNormalizer",
    "WhitespaceNormalizer",
    "build_normalizer",
]

