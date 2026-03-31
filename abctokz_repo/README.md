# Augenblick abctokz — Multilingual Tokenizer

A from-scratch Python tokenizer for **English** and **Devanagari scripts** (Hindi, Marathi, Sindhi), with three model families: **WordLevel**, **BPE**, and **Unigram**. Built for understanding — every design decision is readable in code.

---

## Setup
abctokz is an in-house library and is not available on PyPI. Install directly from source:

```bash
git clone <repo>
cd AUGENBLICK-HACKATHON-2026
pip install -e .                 # core (development install)
pip install -e ".[adapters]"     # + HF tokenizers + SentencePiece
pip install -e ".[dev]"          # + testing tools
```

---

## A Tokenizer in 30 Lines

```python
from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual

# 1. Configure the pipeline
config = bpe_multilingual(vocab_size=8000)

# 2. Train on your corpus
tokenizer = Tokenizer.from_config(config)
tokenizer.train(["data/corpus.txt"], config)
tokenizer.save("artifacts/bpe_tok")

# 3. Reload and use
tok = Tokenizer.load("artifacts/bpe_tok")
enc = tok.encode("नमस्ते world")

print(enc.tokens)          # subword pieces
print(enc.ids)             # vocabulary IDs
print(tok.decode(enc.ids)) # reconstructed text
```

---

## CLI

```bash
abctokz train    --config configs/bpe_multilingual.yaml
abctokz encode   --model artifacts/bpe_tok --text "नमस्ते world"
abctokz decode   --model artifacts/bpe_tok --ids "12,98,44,3"
abctokz inspect  --model artifacts/bpe_tok
abctokz benchmark --config benchmarks/configs/core.yaml
```

---

## What's Inside

The library is intentionally small and readable:

- **Three model families** — WordLevel (frequency lookup), BPE (greedy merge), Unigram (Viterbi decode)
- **Devanagari-first normalizer** — NFC (not NFKC), ZWJ/ZWNJ control, grapheme-cluster integrity
- **Modular pipeline** — normalizer → pre-tokenizer → model → post-processor → decoder
- **Pydantic configs** — frozen, validated, composable
- **Evaluation tooling** — fertility, UNK rate, round-trip fidelity, throughput
- **External adapters** — compare against HuggingFace tokenizers and SentencePiece

---

## Tests

```bash
pytest tests/unit          # component correctness
pytest tests/integration   # full pipeline
pytest tests/golden        # regression (known outputs)
pytest tests/property      # determinism invariants
```

---

## Questions

See [`TASKS.md`](TASKS.md) for the hackathon questions.

---

