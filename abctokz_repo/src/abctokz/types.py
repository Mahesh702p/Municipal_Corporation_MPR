# Augenblick — abctokz
"""Core type definitions and result payloads for the abctokz public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Type aliases
TokenID = int
Token = str
VocabType = dict[str, int]  # token -> id
InverseVocabType = dict[int, str]  # id -> token
MergePair = tuple[str, str]
MergeRules = list[tuple[MergePair, str]]  # [(a, b) -> merged]
PieceScore = tuple[str, float]  # (piece, log-prob)


@dataclass(frozen=True, slots=True)
class Encoding:
    """The output of a :class:`~abctokz.tokenizer.Tokenizer` encode call.

    Attributes:
        ids: Token IDs in vocabulary order.
        tokens: String token representations corresponding to *ids*.
        offsets: ``(start, end)`` character offsets into the **normalized**
            input string for each token.
        special_tokens_mask: 1 for special tokens, 0 otherwise.
        attention_mask: 1 for real tokens, 0 for padding.
        overflowing: Any overflow encodings produced when ``max_length`` is set.
    """

    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]] = field(default_factory=list)
    special_tokens_mask: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    overflowing: list["Encoding"] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return (
            f"Encoding(n_tokens={len(self.ids)}, "
            f"tokens={self.tokens[:8]}{'...' if len(self.tokens) > 8 else ''})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the encoding to a JSON-compatible dict."""
        return {
            "ids": self.ids,
            "tokens": self.tokens,
            "offsets": [list(o) for o in self.offsets],
            "special_tokens_mask": self.special_tokens_mask,
            "attention_mask": self.attention_mask,
        }


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Metadata stored in a tokenizer artifact manifest.

    Attributes:
        schema_version: Version string for backward-compat checks.
        model_type: Model family name (``"wordlevel"``, ``"bpe"``, ``"unigram"``).
        vocab_size: Number of tokens in the vocabulary.
        created_at: ISO-8601 timestamp of when the artifact was created.
        description: Optional human-readable description.
        languages: List of language tags the tokenizer was trained on.
        checksum: SHA-256 hash of the vocab artifact for integrity verification.
        extra: Arbitrary additional metadata.
    """

    schema_version: str
    model_type: str
    vocab_size: int
    created_at: str
    description: str = ""
    languages: list[str] = field(default_factory=list)
    checksum: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "created_at": self.created_at,
            "description": self.description,
            "languages": self.languages,
            "checksum": self.checksum,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactMetadata":
        """Deserialise from a dict."""
        return cls(
            schema_version=data["schema_version"],
            model_type=data["model_type"],
            vocab_size=data["vocab_size"],
            created_at=data["created_at"],
            description=data.get("description", ""),
            languages=data.get("languages", []),
            checksum=data.get("checksum", ""),
            extra=data.get("extra", {}),
        )


@dataclass
class SpecialToken:
    """Represents a special token with its string and ID.

    Attributes:
        content: The string form of the special token (e.g. ``"<unk>"``).
        id: The vocabulary ID.
        single_word: If ``True``, the token is never merged with adjacent chars.
        lstrip: Strip left whitespace when encountered.
        rstrip: Strip right whitespace when encountered.
        normalized: Whether the token is subject to normalization.
    """

    content: str
    id: int
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict."""
        return {
            "content": self.content,
            "id": self.id,
            "single_word": self.single_word,
            "lstrip": self.lstrip,
            "rstrip": self.rstrip,
            "normalized": self.normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpecialToken":
        """Deserialise from dict."""
        return cls(
            content=data["content"],
            id=data["id"],
            single_word=data.get("single_word", False),
            lstrip=data.get("lstrip", False),
            rstrip=data.get("rstrip", False),
            normalized=data.get("normalized", False),
        )


@dataclass
class BenchmarkResult:
    """Results from a single tokenizer benchmark run.

    Attributes:
        tokenizer_name: Identifier of the tokenizer being benchmarked.
        language: Language tag (e.g. ``"hi"``, ``"en"``).
        n_sentences: Number of sentences processed.
        throughput_sps: Sentences per second.
        mean_tokens_per_sentence: Average number of tokens per sentence.
        fertility: Ratio of tokens to reference word count.
        unk_rate: Fraction of tokens that are ``<unk>``.
        round_trip_success_rate: Fraction of sentences that decode identically.
        normalized_seq_length_ratio: Ratio of token count to character count.
        elapsed_seconds: Total elapsed wall-clock time.
    """

    tokenizer_name: str
    language: str
    n_sentences: int
    throughput_sps: float
    mean_tokens_per_sentence: float
    fertility: float
    unk_rate: float
    round_trip_success_rate: float
    normalized_seq_length_ratio: float
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "tokenizer_name": self.tokenizer_name,
            "language": self.language,
            "n_sentences": self.n_sentences,
            "throughput_sps": round(self.throughput_sps, 2),
            "mean_tokens_per_sentence": round(self.mean_tokens_per_sentence, 4),
            "fertility": round(self.fertility, 4),
            "unk_rate": round(self.unk_rate, 6),
            "round_trip_success_rate": round(self.round_trip_success_rate, 6),
            "normalized_seq_length_ratio": round(self.normalized_seq_length_ratio, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "extra": self.extra,
        }
