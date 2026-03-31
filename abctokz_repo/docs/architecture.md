# Architecture

## Pipeline

```
Raw text
  │
  ▼ Normalizer (optional)
Normalized text
  │
  ▼ PreTokenizer (optional)
[pre-token₁, pre-token₂, …]
  │
  ▼ Model (required)
[(token, id), …]
  │
  ▼ PostProcessor (optional)
Encoding
```

## Component Roles

| Component | Role | Stateful? |
|-----------|------|-----------|
| `Normalizer` | Unicode normalization, whitespace collapsing | No |
| `PreTokenizer` | Boundary detection before model sees input | No |
| `Model` | Vocabulary lookup / subword segmentation | Yes (vocab) |
| `Trainer` | Learns model artifacts from corpus | No (produces Model) |
| `PostProcessor` | Adds special tokens (BOS/EOS) | No |
| `Decoder` | Reconstructs text from token list | No |
| `Tokenizer` | Orchestrates the full pipeline | Yes (model + components) |

## Model Families

### WordLevel
- Whole-word vocabulary lookup
- OOV → `<unk>`
- Deterministic by frequency sort + lex tiebreak

### BPE (Byte-Pair Encoding)
- Character-level initialisation with `##` continuation prefix
- Greedy merge by rank
- Deterministic by lex-smallest tiebreak in pair selection

### Unigram
- EM-style training on piece log-probabilities
- Viterbi decoding for best segmentation
- Pruning with character coverage guarantee

## Artifact Layout

```
artifact_dir/
  manifest.json        ← schema_version, model_type, vocab_size, checksum
  config.json          ← model_type (reconstruction hint)
  vocab.json           ← token → id (WordLevel + BPE)
  merges.txt           ← BPE merge rules (BPE only)
  pieces.json          ← [(piece, score), …] (Unigram only)
  special_tokens.json  ← special token registry
```
