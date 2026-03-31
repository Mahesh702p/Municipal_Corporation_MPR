# Augenblick — abctokz
"""Intrinsic evaluation helpers."""

from __future__ import annotations

from abctokz.eval.metrics import (
    fertility,
    mean_tokens_per_sentence,
    normalized_seq_length_ratio,
    round_trip_success_rate,
    unk_rate,
)
from abctokz.tokenizer import Tokenizer
from abctokz.types import BenchmarkResult


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    sentences: list[str],
    name: str,
    language: str = "",
    unk_id: int = 0,
) -> BenchmarkResult:
    """Run intrinsic evaluation of *tokenizer* on *sentences*.

    Computes: fertility, mean tokens/sentence, unk rate, normalized seq-length
    ratio, and round-trip decode success rate.

    Args:
        tokenizer: Trained :class:`~abctokz.tokenizer.Tokenizer` to evaluate.
        sentences: List of evaluation sentences.
        name: Identifier for the result.
        language: Language tag for the result.
        unk_id: ID of the ``<unk>`` token.

    Returns:
        :class:`~abctokz.types.BenchmarkResult`.
    """
    import time

    t0 = time.perf_counter()
    encodings = tokenizer.encode_batch(sentences)
    elapsed = time.perf_counter() - t0

    ref_counts = [len(s.split()) for s in sentences]
    decoded = [tokenizer.decode(enc.ids) for enc in encodings]

    return BenchmarkResult(
        tokenizer_name=name,
        language=language,
        n_sentences=len(sentences),
        throughput_sps=len(sentences) / max(elapsed, 1e-9),
        mean_tokens_per_sentence=mean_tokens_per_sentence(encodings),
        fertility=fertility(encodings, ref_counts),
        unk_rate=unk_rate(encodings, unk_id=unk_id),
        round_trip_success_rate=round_trip_success_rate(sentences, decoded),
        normalized_seq_length_ratio=normalized_seq_length_ratio(encodings, sentences),
        elapsed_seconds=elapsed,
    )
