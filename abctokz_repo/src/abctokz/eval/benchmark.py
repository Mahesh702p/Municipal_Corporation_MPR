# Augenblick — abctokz
"""Benchmark runner: compares multiple tokenizers on corpus samples."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from abctokz.config.schemas import BenchmarkConfig
from abctokz.data.corpus import load_corpus
from abctokz.data.sampling import sample_lines
from abctokz.eval.intrinsic import evaluate_tokenizer
from abctokz.eval.metrics import (
    fertility,
    mean_tokens_per_sentence,
    normalized_seq_length_ratio,
    round_trip_success_rate,
    unk_rate,
)
from abctokz.tokenizer import Tokenizer
from abctokz.types import BenchmarkResult
from abctokz.utils.io import ensure_dir, save_json
from abctokz.utils.logging import get_logger
from abctokz.utils.timer import timed

logger = get_logger(__name__)


class BenchmarkRunner:
    """Run benchmarks comparing multiple tokenizers on corpus data.

    Args:
        config: Benchmark configuration.

    Example::

        from abctokz.config.schemas import BenchmarkConfig
        runner = BenchmarkRunner(BenchmarkConfig(
            name="core",
            corpus_paths=["data/en.txt", "data/hi.txt"],
            tokenizer_paths=["artifacts/bpe_tok", "artifacts/unigram_tok"],
        ))
        results = runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    def run(self) -> list[BenchmarkResult]:
        """Execute the benchmark and return all results.

        Returns:
            List of :class:`~abctokz.types.BenchmarkResult` objects.
        """
        cfg = self._config
        logger.info("Starting benchmark %r", cfg.name)

        # Load and sample corpus
        all_lines = load_corpus(cfg.corpus_paths)
        sentences = sample_lines(all_lines, cfg.sample_size)
        logger.info("Loaded %d sentences from %d files", len(sentences), len(cfg.corpus_paths))

        results: list[BenchmarkResult] = []

        for tok_path in cfg.tokenizer_paths:
            try:
                tokenizer = Tokenizer.load(tok_path)
            except Exception as exc:
                logger.warning("Failed to load tokenizer at %r: %s", tok_path, exc)
                continue

            name = Path(tok_path).name
            lang = cfg.languages[0] if cfg.languages else ""

            # Warmup
            for _ in range(cfg.warmup_runs):
                tokenizer.encode_batch(sentences[:10])

            # Timed runs
            all_encodings = []
            total_elapsed = 0.0
            for _ in range(cfg.timed_runs):
                with timed() as t:
                    encodings = tokenizer.encode_batch(sentences)
                total_elapsed += t["elapsed"]
                all_encodings = encodings  # keep last run for metrics

            avg_elapsed = total_elapsed / cfg.timed_runs
            decoded = [tokenizer.decode(enc.ids) for enc in all_encodings]
            ref_counts = [len(s.split()) for s in sentences]

            result = BenchmarkResult(
                tokenizer_name=name,
                language=lang,
                n_sentences=len(sentences),
                throughput_sps=len(sentences) / max(avg_elapsed, 1e-9),
                mean_tokens_per_sentence=mean_tokens_per_sentence(all_encodings),
                fertility=fertility(all_encodings, ref_counts),
                unk_rate=unk_rate(all_encodings),
                round_trip_success_rate=round_trip_success_rate(sentences, decoded),
                normalized_seq_length_ratio=normalized_seq_length_ratio(all_encodings, sentences),
                elapsed_seconds=avg_elapsed,
            )
            results.append(result)
            logger.info(
                "Benchmarked %r: fertility=%.3f, unk_rate=%.4f, throughput=%.1f sps",
                name, result.fertility, result.unk_rate, result.throughput_sps,
            )

        return results

    def save_results(self, results: list[BenchmarkResult]) -> dict[str, str]:
        """Save results as JSON and Markdown.

        Args:
            results: Benchmark results to save.

        Returns:
            Dict with ``"json"`` and ``"markdown"`` paths.
        """
        from abctokz.eval.reports import results_to_markdown

        out_dir = ensure_dir(self._config.output_dir)
        safe_name = self._config.name.replace(" ", "_")

        json_path = out_dir / f"{safe_name}.json"
        md_path = out_dir / f"{safe_name}.md"

        save_json([r.to_dict() for r in results], json_path)
        md_path.write_text(results_to_markdown(results, title=self._config.name), encoding="utf-8")

        logger.info("Benchmark results saved to %s", out_dir)
        return {"json": str(json_path), "markdown": str(md_path)}
