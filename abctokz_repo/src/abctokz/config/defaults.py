# Augenblick — abctokz
"""Pre-built configuration presets for common abctokz use cases."""

from __future__ import annotations

from abctokz.config.schemas import (
    BPEConfig,
    BPETrainerConfig,
    DevanagariNormalizerConfig,
    DevanagariAwarePreTokenizerConfig,
    NfkcNormalizerConfig,
    SequenceNormalizerConfig,
    SequencePreTokenizerConfig,
    TokenizerConfig,
    UnigramConfig,
    UnigramTrainerConfig,
    WhitespaceNormalizerConfig,
    WhitespacePreTokenizerConfig,
    WordLevelConfig,
    WordLevelTrainerConfig,
)
from abctokz.constants import UNK_TOKEN


def english_basic_normalizer() -> SequenceNormalizerConfig:
    """Normalizer preset suitable for plain English text.

    Applies NFKC normalization followed by whitespace collapsing.

    Returns:
        :class:`~abctokz.config.schemas.SequenceNormalizerConfig`.
    """
    return SequenceNormalizerConfig(
        normalizers=[
            NfkcNormalizerConfig(strip_zero_width=True),
            WhitespaceNormalizerConfig(strip=True, collapse=True),
        ]
    )


def devanagari_safe_normalizer() -> SequenceNormalizerConfig:
    """Normalizer preset for Devanagari text (Hindi / Marathi / Sindhi).

    Uses NFC (not NFKC) to avoid lossy normalization of Devanagari combining
    marks, then applies Devanagari-safe rules.

    Returns:
        :class:`~abctokz.config.schemas.SequenceNormalizerConfig`.
    """
    return SequenceNormalizerConfig(
        normalizers=[
            DevanagariNormalizerConfig(nfc_first=True, strip_zero_width=False),
            WhitespaceNormalizerConfig(strip=True, collapse=True),
        ]
    )


def multilingual_shared_normalizer() -> SequenceNormalizerConfig:
    """Normalizer for mixed English + Devanagari text.

    Applies NFC (preserves Devanagari combining marks) and whitespace
    normalization without NFKC (which would mangle some Devanagari).

    Returns:
        :class:`~abctokz.config.schemas.SequenceNormalizerConfig`.
    """
    return SequenceNormalizerConfig(
        normalizers=[
            DevanagariNormalizerConfig(nfc_first=True, strip_zero_width=False),
            WhitespaceNormalizerConfig(strip=True, collapse=True),
        ]
    )


def wordlevel_multilingual(vocab_size: int = 8_000) -> TokenizerConfig:
    """Default WordLevel tokenizer config for multilingual (EN + Devanagari).

    Args:
        vocab_size: Target vocabulary size.

    Returns:
        :class:`~abctokz.config.schemas.TokenizerConfig`.
    """
    return TokenizerConfig(
        normalizer=multilingual_shared_normalizer(),
        pretokenizer=SequencePreTokenizerConfig(
            pretokenizers=[
                DevanagariAwarePreTokenizerConfig(
                    split_on_whitespace=True, split_on_script_boundary=True
                ),
            ]
        ),
        model=WordLevelConfig(unk_token=UNK_TOKEN, vocab_size=vocab_size),
        trainer=WordLevelTrainerConfig(
            vocab_size=vocab_size, min_frequency=2, special_tokens=[UNK_TOKEN]
        ),
    )


def bpe_multilingual(vocab_size: int = 8_000) -> TokenizerConfig:
    """Default BPE tokenizer config for multilingual (EN + Devanagari).

    Args:
        vocab_size: Target vocabulary size.

    Returns:
        :class:`~abctokz.config.schemas.TokenizerConfig`.
    """
    return TokenizerConfig(
        normalizer=multilingual_shared_normalizer(),
        pretokenizer=SequencePreTokenizerConfig(
            pretokenizers=[
                DevanagariAwarePreTokenizerConfig(
                    split_on_whitespace=True, split_on_script_boundary=True
                ),
            ]
        ),
        model=BPEConfig(unk_token=UNK_TOKEN, vocab_size=vocab_size),
        trainer=BPETrainerConfig(
            vocab_size=vocab_size, min_frequency=2, special_tokens=[UNK_TOKEN]
        ),
    )


def unigram_multilingual(vocab_size: int = 8_000) -> TokenizerConfig:
    """Default Unigram tokenizer config for multilingual (EN + Devanagari).

    Args:
        vocab_size: Target vocabulary size.

    Returns:
        :class:`~abctokz.config.schemas.TokenizerConfig`.
    """
    return TokenizerConfig(
        normalizer=multilingual_shared_normalizer(),
        pretokenizer=SequencePreTokenizerConfig(
            pretokenizers=[
                DevanagariAwarePreTokenizerConfig(
                    split_on_whitespace=True, split_on_script_boundary=True
                ),
            ]
        ),
        model=UnigramConfig(unk_token=UNK_TOKEN, vocab_size=vocab_size),
        trainer=UnigramTrainerConfig(
            vocab_size=vocab_size, special_tokens=[UNK_TOKEN]
        ),
    )
