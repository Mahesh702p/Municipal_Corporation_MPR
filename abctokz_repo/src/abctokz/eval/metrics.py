# Augenblick — abctokz
"""Tokenizer evaluation metrics."""

from __future__ import annotations

from abctokz.types import BenchmarkResult, Encoding


def fertility(encodings: list[Encoding], reference_word_counts: list[int]) -> float:
    """Compute the fertility score: mean tokens-per-reference-word.

    Fertility measures how many subword tokens a tokenizer produces per
    reference word.  Lower is better for efficiency.

    Args:
        encodings: List of encodings (one per sentence).
        reference_word_counts: Number of whitespace-split words per sentence.

    Returns:
        Mean fertility score.
    """
    if not encodings:
        return 0.0
    total_tokens = sum(len(e) for e in encodings)
    total_ref = sum(reference_word_counts)
    if total_ref == 0:
        return 0.0
    return total_tokens / total_ref


def unk_rate(encodings: list[Encoding], unk_id: int = 0) -> float:
    """Compute the unknown-token rate.

    Args:
        encodings: List of encodings.
        unk_id: The vocabulary ID of the ``<unk>`` token.

    Returns:
        Fraction of total tokens that are ``<unk>``.
    """
    total = sum(len(e) for e in encodings)
    if total == 0:
        return 0.0
    n_unk = sum(e.ids.count(unk_id) for e in encodings)
    return n_unk / total


def mean_tokens_per_sentence(encodings: list[Encoding]) -> float:
    """Compute mean number of tokens per sentence.

    Args:
        encodings: List of encodings.

    Returns:
        Mean token count per encoding.
    """
    if not encodings:
        return 0.0
    return sum(len(e) for e in encodings) / len(encodings)


def normalized_seq_length_ratio(encodings: list[Encoding], texts: list[str]) -> float:
    """Compute mean tokens-per-character ratio.

    Args:
        encodings: List of encodings.
        texts: Original texts (same order as encodings).

    Returns:
        Mean (n_tokens / n_chars) ratio.
    """
    ratios = []
    for enc, text in zip(encodings, texts):
        n_chars = len(text)
        if n_chars > 0:
            ratios.append(len(enc) / n_chars)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def round_trip_success_rate(
    originals: list[str],
    decoded: list[str],
    normalized_originals: list[str] | None = None,
) -> float:
    """Compute the fraction of sentences that survive the encode-decode round trip.

    Two strings are considered equivalent if they match exactly.  If
    *normalized_originals* is provided, decoded strings are compared against
    those (useful when normalization is lossless relative to the normalized
    form rather than the raw form).

    Args:
        originals: Raw input sentences.
        decoded: Decoded sentences from the tokenizer.
        normalized_originals: Optional normalized versions of *originals* to
            compare against.

    Returns:
        Fraction of successful round trips.
    """
    targets = normalized_originals if normalized_originals is not None else originals
    if not targets:
        return 0.0
    matches = sum(1 for t, d in zip(targets, decoded) if t == d)
    return matches / len(targets)
