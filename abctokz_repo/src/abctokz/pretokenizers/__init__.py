# Augenblick — abctokz
"""Pre-tokenizer subpackage for abctokz."""

from abctokz.pretokenizers.base import PreTokenizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer
from abctokz.pretokenizers.punctuation import PunctuationPreTokenizer
from abctokz.pretokenizers.regex import RegexPreTokenizer
from abctokz.pretokenizers.sequence import SequencePreTokenizer
from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer
from abctokz.config.schemas import (
    AnyPreTokenizerConfig,
    DevanagariAwarePreTokenizerConfig,
    PunctuationPreTokenizerConfig,
    RegexPreTokenizerConfig,
    SequencePreTokenizerConfig,
    WhitespacePreTokenizerConfig,
)


def build_pretokenizer(config: AnyPreTokenizerConfig) -> PreTokenizer:
    """Construct a :class:`PreTokenizer` from a config object.

    Args:
        config: A validated pre-tokenizer config.

    Returns:
        Corresponding :class:`PreTokenizer` instance.

    Raises:
        ValueError: For unknown config types.
    """
    if isinstance(config, WhitespacePreTokenizerConfig):
        return WhitespacePreTokenizer()
    if isinstance(config, PunctuationPreTokenizerConfig):
        return PunctuationPreTokenizer(behavior=config.behavior)
    if isinstance(config, RegexPreTokenizerConfig):
        return RegexPreTokenizer(pattern=config.pattern, invert=config.invert)
    if isinstance(config, DevanagariAwarePreTokenizerConfig):
        return DevanagariAwarePreTokenizer(
            split_on_whitespace=config.split_on_whitespace,
            split_on_script_boundary=config.split_on_script_boundary,
        )
    if isinstance(config, SequencePreTokenizerConfig):
        return SequencePreTokenizer([build_pretokenizer(c) for c in config.pretokenizers])
    raise ValueError(f"Unknown pre-tokenizer config type: {type(config)}")


__all__ = [
    "PreTokenizer",
    "DevanagariAwarePreTokenizer",
    "PunctuationPreTokenizer",
    "RegexPreTokenizer",
    "SequencePreTokenizer",
    "WhitespacePreTokenizer",
    "build_pretokenizer",
]

