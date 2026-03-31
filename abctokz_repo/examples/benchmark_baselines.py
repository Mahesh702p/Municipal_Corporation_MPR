"""Example: benchmark abctokz models against each other.

Run::

    python examples/benchmark_baselines.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
from abctokz.eval.benchmark import BenchmarkRunner
from abctokz.eval.intrinsic import evaluate_tokenizer
from abctokz.tokenizer import Tokenizer

CORPUS_LINES = [
    "hello world",
    "the quick brown fox",
    "नमस्ते दुनिया",
    "हिन्दी भाषा में टोकनाइजेशन",
    "मराठी भाषेत टोकनायझेशन",
    "सिन्धी भाषा",
    "hello नमस्ते world दुनिया",
    "Devanagari script नागरी लिपि",
] * 30

EVAL_SENTENCES = [
    "hello world",
    "नमस्ते दुनिया",
    "मराठी भाषेत परीक्षण",
    "सिन्धी भाषा का परीक्षण",
    "hello नमस्ते world दुनिया",
    "BPE tokenizer for multilingual text",
    "हिन्दी और अंग्रेज़ी मिश्रित वाक्य",
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        tokenizers: dict[str, Tokenizer] = {}

        for model_type, config_fn in [
            ("wordlevel", wordlevel_multilingual),
            ("bpe", bpe_multilingual),
            ("unigram", unigram_multilingual),
        ]:
            config = config_fn(vocab_size=200)
            tok = Tokenizer.from_config(config)
            tok.train([str(corpus_path)], config)
            tokenizers[model_type] = tok
            print(f"Trained {model_type}: vocab_size={tok.get_vocab_size()}")

        print("\n--- Benchmark Results ---\n")
        results = []
        for name, tokenizer in tokenizers.items():
            result = evaluate_tokenizer(
                tokenizer, EVAL_SENTENCES, name=name, language="mixed"
            )
            results.append(result)
            print(f"{name}:")
            print(f"  Fertility:          {result.fertility:.3f}")
            print(f"  Mean tokens/sent:   {result.mean_tokens_per_sentence:.2f}")
            print(f"  UNK rate:           {result.unk_rate:.4f}")
            print(f"  Round-trip success: {result.round_trip_success_rate * 100:.1f}%")
            print(f"  Seq-length ratio:   {result.normalized_seq_length_ratio:.3f}")
            print(f"  Throughput:         {result.throughput_sps:.1f} sps")
            print()

        # Print Markdown table
        from abctokz.eval.reports import results_to_markdown
        print(results_to_markdown(results, title="Multilingual Tokenizer Benchmark"))


if __name__ == "__main__":
    main()
