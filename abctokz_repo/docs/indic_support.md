# Indic Language Support

## Supported Scripts

All three Devanagari-script languages are supported:

| Language | Script | Notes |
|----------|--------|-------|
| Hindi (hi) | Devanagari | Core support, tested |
| Marathi (mr) | Devanagari | Same script as Hindi, tested |
| Sindhi (sd) | Devanagari | Devanagari variant, tested |
| English (en) | Latin | Core support |

## Normalization Policy

- **Use NFC** (not NFKC) for Devanagari text. NFKC can be lossy for some
  Devanagari combining marks.
- **Zero-width characters** (ZWJ U+200D, ZWNJ U+200C) are preserved by default
  in Devanagari mode because they carry conjunct-formation semantics.
  Set `strip_zero_width=True` in `DevanagariNormalizerConfig` to remove them.
- **Losslessness**: decoding is lossless relative to the *normalized* input,
  not the raw input. If you need raw-input losslessness, use `IdentityNormalizer`.

## Pre-tokenization

`DevanagariAwarePreTokenizer` splits on:
1. Whitespace (always, if `split_on_whitespace=True`).
2. Script boundaries — transitions between Devanagari and Latin within a word
   (if `split_on_script_boundary=True`).

Grapheme cluster integrity is preserved: matras, halant, chandrabindu,
anusvara, and visarga remain attached to their base consonants.

## Fertility Metrics

Fertility (tokens / reference words) is reported per language in benchmarks.
For agglutinative or morphologically rich languages, higher fertility is
expected with smaller vocabulary sizes. Use the `fertility` metric to compare
tokenizer efficiency across language-vocab size combinations.

## Recommended Configs

```python
from abctokz.config.defaults import (
    devanagari_safe_normalizer,
    multilingual_shared_normalizer,
    bpe_multilingual,
    unigram_multilingual,
)

# For pure Devanagari text:
norm = devanagari_safe_normalizer()

# For mixed EN + Devanagari:
norm = multilingual_shared_normalizer()
config = bpe_multilingual(vocab_size=8000)
```
