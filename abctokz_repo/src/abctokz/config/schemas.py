# Augenblick — abctokz
"""Pydantic config schemas for all abctokz components."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from abctokz.constants import (
    BPE_CONTINUATION_PREFIX,
    BPE_DEFAULT_MIN_FREQUENCY,
    BPE_DEFAULT_VOCAB_SIZE,
    DEFAULT_VOCAB_SIZE,
    EOS_TOKEN,
    MIN_FREQUENCY,
    UNK_TOKEN,
    UNIGRAM_CHAR_COVERAGE,
    UNIGRAM_DEFAULT_VOCAB_SIZE,
    UNIGRAM_NUM_SUB_ITERATIONS,
    UNIGRAM_SHRINKING_FACTOR,
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseConfig(BaseModel):
    """Base config with strict behaviour."""

    model_config = {"extra": "forbid", "frozen": True}


# ---------------------------------------------------------------------------
# Normalizer configs
# ---------------------------------------------------------------------------


class IdentityNormalizerConfig(BaseConfig):
    """Pass-through normalizer; no transformation applied."""

    type: Literal["identity"] = "identity"


class NfkcNormalizerConfig(BaseConfig):
    """Unicode NFKC normalization."""

    type: Literal["nfkc"] = "nfkc"
    strip_zero_width: bool = Field(default=True, description="Remove zero-width chars after NFKC.")


class WhitespaceNormalizerConfig(BaseConfig):
    """Collapse/normalise whitespace."""

    type: Literal["whitespace"] = "whitespace"
    strip: bool = Field(default=True, description="Strip leading/trailing whitespace.")
    collapse: bool = Field(default=True, description="Collapse multiple spaces into one.")


class DevanagariNormalizerConfig(BaseConfig):
    """Devanagari-safe normalization."""

    type: Literal["devanagari"] = "devanagari"
    nfc_first: bool = Field(default=True, description="Apply NFC before Devanagari rules.")
    strip_zero_width: bool = Field(default=False, description="Remove ZWJ/ZWNJ characters.")


NormalizerConfig = (
    IdentityNormalizerConfig
    | NfkcNormalizerConfig
    | WhitespaceNormalizerConfig
    | DevanagariNormalizerConfig
)


class SequenceNormalizerConfig(BaseConfig):
    """Chain multiple normalizers in order."""

    type: Literal["sequence"] = "sequence"
    normalizers: list[NormalizerConfig] = Field(default_factory=list)


AnyNormalizerConfig = NormalizerConfig | SequenceNormalizerConfig


# ---------------------------------------------------------------------------
# Pre-tokenizer configs
# ---------------------------------------------------------------------------


class WhitespacePreTokenizerConfig(BaseConfig):
    """Split on whitespace."""

    type: Literal["whitespace"] = "whitespace"


class PunctuationPreTokenizerConfig(BaseConfig):
    """Split on punctuation boundaries."""

    type: Literal["punctuation"] = "punctuation"
    behavior: Literal["isolated", "merged_with_previous", "merged_with_next"] = "isolated"


class RegexPreTokenizerConfig(BaseConfig):
    """Split using a custom regex pattern."""

    type: Literal["regex"] = "regex"
    pattern: str = Field(..., description="Regex pattern for splitting.")
    invert: bool = Field(default=False, description="If True, keep matches; otherwise split on them.")


class DevanagariAwarePreTokenizerConfig(BaseConfig):
    """Splits that respect Devanagari grapheme boundaries."""

    type: Literal["devanagari_aware"] = "devanagari_aware"
    split_on_whitespace: bool = Field(default=True)
    split_on_script_boundary: bool = Field(default=True, description="Split between scripts.")


PreTokenizerConfig = (
    WhitespacePreTokenizerConfig
    | PunctuationPreTokenizerConfig
    | RegexPreTokenizerConfig
    | DevanagariAwarePreTokenizerConfig
)


class SequencePreTokenizerConfig(BaseConfig):
    """Chain multiple pre-tokenizers in order."""

    type: Literal["sequence"] = "sequence"
    pretokenizers: list[PreTokenizerConfig] = Field(default_factory=list)


AnyPreTokenizerConfig = PreTokenizerConfig | SequencePreTokenizerConfig


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


class WordLevelConfig(BaseConfig):
    """WordLevel model config."""

    type: Literal["wordlevel"] = "wordlevel"
    unk_token: str = Field(default=UNK_TOKEN)
    vocab_size: int = Field(default=DEFAULT_VOCAB_SIZE, ge=1)


class BPEConfig(BaseConfig):
    """BPE model config."""

    type: Literal["bpe"] = "bpe"
    unk_token: str = Field(default=UNK_TOKEN)
    vocab_size: int = Field(default=BPE_DEFAULT_VOCAB_SIZE, ge=1)
    continuation_prefix: str = Field(
        default=BPE_CONTINUATION_PREFIX,
        description="Prefix added to non-initial subword pieces.",
    )
    end_of_word_suffix: str = Field(default="", description="Suffix added to the last piece of a word.")


class UnigramConfig(BaseConfig):
    """Unigram model config."""

    type: Literal["unigram"] = "unigram"
    unk_token: str = Field(default=UNK_TOKEN)
    vocab_size: int = Field(default=UNIGRAM_DEFAULT_VOCAB_SIZE, ge=1)
    unk_id: int = Field(default=0, ge=0)


ModelConfig = WordLevelConfig | BPEConfig | UnigramConfig


# ---------------------------------------------------------------------------
# Trainer configs
# ---------------------------------------------------------------------------


class WordLevelTrainerConfig(BaseConfig):
    """WordLevel trainer config."""

    type: Literal["wordlevel"] = "wordlevel"
    vocab_size: int = Field(default=DEFAULT_VOCAB_SIZE, ge=1)
    min_frequency: int = Field(default=MIN_FREQUENCY, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: [UNK_TOKEN])
    show_progress: bool = True
    seed: int = Field(default=42, description="Random seed for determinism.")


class BPETrainerConfig(BaseConfig):
    """BPE trainer config."""

    type: Literal["bpe"] = "bpe"
    vocab_size: int = Field(default=BPE_DEFAULT_VOCAB_SIZE, ge=1)
    min_frequency: int = Field(default=BPE_DEFAULT_MIN_FREQUENCY, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: [UNK_TOKEN])
    limit_alphabet: Optional[int] = Field(default=None, description="Max initial alphabet size.")
    initial_alphabet: list[str] = Field(default_factory=list)
    continuing_subword_prefix: str = Field(default=BPE_CONTINUATION_PREFIX)
    end_of_word_suffix: str = Field(default="")
    show_progress: bool = True
    seed: int = Field(default=42)


class UnigramTrainerConfig(BaseConfig):
    """Unigram trainer config."""

    type: Literal["unigram"] = "unigram"
    vocab_size: int = Field(default=UNIGRAM_DEFAULT_VOCAB_SIZE, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: [UNK_TOKEN])
    shrinking_factor: float = Field(default=UNIGRAM_SHRINKING_FACTOR, gt=0.0, lt=1.0)
    unk_token: str = Field(default=UNK_TOKEN)
    max_piece_length: int = Field(default=16, ge=1)
    n_sub_iterations: int = Field(default=UNIGRAM_NUM_SUB_ITERATIONS, ge=1)
    char_coverage: float = Field(default=UNIGRAM_CHAR_COVERAGE, gt=0.0, le=1.0)
    show_progress: bool = True
    seed: int = Field(default=42)


TrainerConfig = WordLevelTrainerConfig | BPETrainerConfig | UnigramTrainerConfig


# ---------------------------------------------------------------------------
# Top-level tokenizer config
# ---------------------------------------------------------------------------


class TokenizerConfig(BaseConfig):
    """Full tokenizer configuration bundling all pipeline stages."""

    schema_version: str = Field(default="1", description="Artifact schema version.")
    normalizer: Optional[AnyNormalizerConfig] = None
    pretokenizer: Optional[AnyPreTokenizerConfig] = None
    model: ModelConfig
    trainer: Optional[TrainerConfig] = None
    add_bos: bool = Field(default=False, description="Prepend BOS token during post-processing.")
    add_eos: bool = Field(default=False, description="Append EOS token during post-processing.")
    bos_token: str = Field(default="<s>")
    eos_token: str = Field(default=EOS_TOKEN)
    pad_token: str = Field(default="<pad>")

    @model_validator(mode="after")
    def check_trainer_model_alignment(self) -> "TokenizerConfig":
        """Verify trainer and model types are compatible when both are set."""
        if self.trainer is not None and self.model.type != self.trainer.type:
            raise ValueError(
                f"Model type '{self.model.type}' and trainer type '{self.trainer.type}' must match."
            )
        return self


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


class BenchmarkConfig(BaseConfig):
    """Configuration for a benchmark run."""

    name: str = Field(..., description="Human-readable benchmark name.")
    corpus_paths: list[str] = Field(..., description="Paths to text corpora.")
    tokenizer_paths: list[str] = Field(..., description="Paths to saved tokenizer artifacts.")
    sample_size: int = Field(default=1_000, ge=1)
    warmup_runs: int = Field(default=3, ge=0)
    timed_runs: int = Field(default=10, ge=1)
    output_dir: str = Field(default="benchmarks/outputs")
    languages: list[str] = Field(default_factory=list, description="Language tags, e.g. ['en', 'hi'].")


# ---------------------------------------------------------------------------
# Training run config (used by CLI)
# ---------------------------------------------------------------------------


class TrainingRunConfig(BaseModel):
    """Top-level config for a abctokz train CLI run."""

    model_config = {"extra": "forbid"}

    output_dir: str = Field(..., description="Where to save the trained tokenizer artifact.")
    corpus: list[str] = Field(..., description="Paths to training corpus files.")
    tokenizer: TokenizerConfig

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return self.model_dump()
