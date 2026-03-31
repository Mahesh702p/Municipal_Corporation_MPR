# Augenblick — abctokz
"""Custom exceptions for abctokz."""

from __future__ import annotations


class TokzError(Exception):
    """Base exception for all abctokz errors."""


class VocabError(TokzError):
    """Raised for vocabulary-related errors (OOV, size mismatches, etc.)."""


class TrainingError(TokzError):
    """Raised when tokenizer training fails or produces invalid state."""


class SerializationError(TokzError):
    """Raised for artifact save/load failures or schema mismatches."""


class SchemaVersionError(SerializationError):
    """Raised when a saved artifact has an incompatible schema version."""

    def __init__(self, found: str, expected: str) -> None:
        super().__init__(
            f"Incompatible schema version: found '{found}', expected '{expected}'."
            " Please retrain or migrate the artifact."
        )
        self.found = found
        self.expected = expected


class ConfigError(TokzError):
    """Raised for invalid or inconsistent configuration."""


class NormalizationError(TokzError):
    """Raised when normalization fails."""


class PreTokenizationError(TokzError):
    """Raised when pre-tokenization fails."""


class DecodingError(TokzError):
    """Raised when decoding fails or produces inconsistent output."""


class AdapterError(TokzError):
    """Raised when an external tokenizer adapter fails."""


class BenchmarkError(TokzError):
    """Raised when benchmark execution fails."""


class UnknownTokenError(VocabError):
    """Raised when a token is not in the vocabulary and no fallback exists."""

    def __init__(self, token: str) -> None:
        super().__init__(f"Token not in vocabulary: {token!r}")
        self.token = token
