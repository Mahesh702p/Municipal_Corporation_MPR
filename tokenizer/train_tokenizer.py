"""
train_tokenizer.py
==================
Train and benchmark the MunicipalTokenizer on the collected corpus.
Run this AFTER clean_pipeline.py has generated pretrain_corpus.txt.

Outputs:
  artifacts/municipal_bpe_tok/      ← trained tokenizer artifact
  artifacts/tokenizer_benchmark.txt ← fertility, UNK rate, throughput

Usage: python tokenizer/train_tokenizer.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure abctokz and project root are on path
sys.path.insert(0, str(Path(__file__).parent.parent / "abctokz_repo" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.municipal_tokenizer import MunicipalTokenizer

CORPUS_PATH = "data/processed/pretrain_corpus.txt"
SAVE_PATH = "artifacts/municipal_bpe_tok"
VOCAB_SIZE = 10000
MAX_LEN = 60
MODEL_TYPE = "bpe"

# Test sentences (EN + Hinglish + Devanagari)
TEST_SENTENCES = [
    "garbage not collected in ward 5 since 3 days",
    "road pothole near market junction causing accidents",
    "paani nahi aa raha 3 din se please help",
    "property tax demand notice received but payment already done",
    "kachra nahi utha ward 12 mein",
    "पानी नहीं आ रहा है तीन दिनों से",
    "सड़क पर बड़ा गड्ढा है जिससे दुर्घटना का खतरा है",
    "street light not working near school road after 8pm",
    "building construction going on without permission in plot 45",
    "drain blocked causing water logging in colony",
]


def benchmark_tokenizer(tok: MunicipalTokenizer):
    """Evaluate tokenizer quality metrics."""
    results = []
    total_words = 0
    total_tokens = 0
    unk_count = 0
    unk_id = tok.get_special_token_id("<UNK>")

    for text in TEST_SENTENCES:
        ids = tok.encode(text)
        tokens = ids  # we count token length
        words = text.split()
        fertility = len(ids) / max(len(words), 1)  # tokens per word
        unk_in_seq = sum(1 for i in ids if i == unk_id)

        results.append({
            "text": text,
            "words": len(words),
            "tokens": len(ids),
            "fertility": round(fertility, 2),
            "unk_count": unk_in_seq,
        })
        total_words += len(words)
        total_tokens += len(ids)
        unk_count += unk_in_seq

    overall_fertility = total_tokens / max(total_words, 1)
    unk_rate = unk_count / max(total_tokens, 1)

    return results, overall_fertility, unk_rate


def main():
    os.makedirs("artifacts", exist_ok=True)

    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        print("Run: python preprocessing/clean_pipeline.py first")
        return

    corpus_size = sum(1 for _ in open(CORPUS_PATH, encoding="utf-8"))
    print(f"Corpus: {corpus_size:,} lines → {CORPUS_PATH}")

    # Train
    print(f"\nTraining MunicipalTokenizer (model={MODEL_TYPE}, vocab_size={VOCAB_SIZE})...")
    tok = MunicipalTokenizer(vocab_size=VOCAB_SIZE, model_type=MODEL_TYPE, max_len=MAX_LEN)

    t_start = time.time()
    tok.train([CORPUS_PATH])
    t_train = time.time() - t_start
    print(f"Training time: {t_train:.1f}s")
    print(f"Final vocab size: {tok.get_vocab_size():,}")

    # Save
    tok.save(SAVE_PATH)

    # Benchmark
    print("\n--- Tokenizer Benchmark ---")
    results, fertility, unk_rate = benchmark_tokenizer(tok)

    bench_lines = [
        "MunicipalTokenizer Benchmark",
        "=" * 50,
        f"Model type:    {MODEL_TYPE}",
        f"Vocab size:    {tok.get_vocab_size():,}",
        f"Training time: {t_train:.1f}s",
        f"Corpus lines:  {corpus_size:,}",
        "",
        f"Overall fertility (tokens/word): {fertility:.3f}",
        f"UNK rate: {unk_rate:.4f} ({unk_rate*100:.2f}%)",
        "",
        "Per-sentence results:",
        "-" * 50,
    ]

    for r in results:
        line = (f"  '{r['text'][:50]:50s}' | "
                f"words={r['words']:3d} tokens={r['tokens']:3d} "
                f"fertility={r['fertility']:.2f} unk={r['unk_count']}")
        bench_lines.append(line)
        print(line)

    bench_lines.extend([
        "",
        "Encode-batch test:",
        f"  Input: {len(TEST_SENTENCES)} sentences",
    ])

    # Test encode_batch (TF-ready)
    import numpy as np
    padded = tok.encode_batch(TEST_SENTENCES, max_len=MAX_LEN)
    bench_lines.append(f"  Output shape: {padded.shape} dtype={padded.dtype}")
    bench_lines.append(f"  PAD zeros: {(padded == 0).sum()} / {padded.size}")
    print(f"\nencode_batch output shape: {padded.shape}, dtype: {padded.dtype}")

    bench_path = "artifacts/tokenizer_benchmark.txt"
    with open(bench_path, "w") as f:
        f.write("\n".join(bench_lines))

    print(f"\n✓ Tokenizer saved: {SAVE_PATH}/")
    print(f"✓ Benchmark saved: {bench_path}")

    # Quality guidance
    print("\n--- Quality Guidance ---")
    if fertility < 1.5:
        print("✓ Excellent fertility (close to 1.0 — mostly whole words)")
    elif fertility < 2.5:
        print("✓ Good fertility (some subword splits — healthy for BPE)")
    else:
        print("⚠ High fertility — consider increasing vocab_size")

    if unk_rate < 0.01:
        print("✓ Excellent UNK rate (< 1%)")
    elif unk_rate < 0.05:
        print("✓ Acceptable UNK rate (< 5%)")
    else:
        print("⚠ High UNK rate — add more domain text to corpus")


if __name__ == "__main__":
    main()
