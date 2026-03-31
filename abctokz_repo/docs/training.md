# Training Guide

## WordLevel

```python
from abctokz.config.defaults import wordlevel_multilingual
from abctokz.tokenizer import Tokenizer

config = wordlevel_multilingual(vocab_size=8000)
tokenizer = Tokenizer.from_config(config)
tokenizer.train(["data/corpus.txt"], config)
tokenizer.save("artifacts/wordlevel_tok")
```

Training is **O(N)** in corpus size. Ordering:
1. Count word frequencies.
2. Filter by `min_frequency`.
3. Assign IDs: special tokens first, then words sorted by (freq↓, lex↑).

## BPE

```python
from abctokz.config.defaults import bpe_multilingual
from abctokz.tokenizer import Tokenizer

config = bpe_multilingual(vocab_size=8000)
tokenizer = Tokenizer.from_config(config)
tokenizer.train(["data/corpus.txt"], config)
tokenizer.save("artifacts/bpe_tok")
```

Training complexity is approximately **O(N × V)** where N = corpus size, V = merges.
Tie-breaking uses lexicographic order for reproducibility.

## Unigram

```python
from abctokz.config.defaults import unigram_multilingual
from abctokz.tokenizer import Tokenizer

config = unigram_multilingual(vocab_size=8000)
tokenizer = Tokenizer.from_config(config)
tokenizer.train(["data/corpus.txt"], config)
tokenizer.save("artifacts/unigram_tok")
```

The EM training loop:
1. Build seed vocabulary from all character n-grams up to `max_piece_length`.
2. E-step: Viterbi-segment corpus, accumulate counts.
3. M-step: Recompute log-probs; prune to `vocab_size × shrinking_factor`.
4. Repeat until `vocab_size` is reached.

## Determinism

All trainers accept a `seed` parameter (default `42`). The same corpus +
same seed always produces identical artifacts. Set different seeds to explore
vocabulary variation.
