# CLI Reference

## Installation

```bash
pip install -e ".[dev]"
abctokz --help
```

## Commands

### `abctokz train`

Train a tokenizer from a YAML config or inline options.

```bash
# From YAML config
abctokz train --config configs/bpe_hi_mr_sd_en.yaml

# Inline
abctokz train --corpus data/hi.txt data/en.txt \
           --model bpe \
           --vocab-size 8000 \
           --output artifacts/bpe_tok
```

Config YAML format:
```yaml
output_dir: artifacts/bpe_tok
corpus:
  - data/hi.txt
  - data/en.txt
tokenizer:
  schema_version: "1"
  model:
    type: bpe
    vocab_size: 8000
  trainer:
    type: bpe
    vocab_size: 8000
    min_frequency: 2
    special_tokens: ["<unk>"]
```

### `abctokz encode`

```bash
abctokz encode --model artifacts/bpe_tok --text "नमस्ते world"
abctokz encode --model artifacts/bpe_tok --input sentences.txt --ids
abctokz encode --model artifacts/bpe_tok --text "hello" --offsets
```

### `abctokz decode`

```bash
abctokz decode --model artifacts/bpe_tok --ids "12,98,44,3"
abctokz decode --model artifacts/bpe_tok --input ids.txt
```

### `abctokz inspect`

```bash
abctokz inspect --model artifacts/bpe_tok
abctokz inspect --model artifacts/bpe_tok --vocab --top-n 50
```

### `abctokz benchmark`

```bash
abctokz benchmark --config benchmarks/configs/core.yaml
abctokz benchmark --corpus data/hi.txt --model artifacts/bpe --model artifacts/unigram \
               --sample-size 500 --name my_bench
```
