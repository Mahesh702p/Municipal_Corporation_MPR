# Augenblick — abctokz
"""abctokz — A from-scratch multilingual tokenizer library.

Supports English and Devanagari scripts (Hindi, Marathi, Sindhi) with
three native model families: WordLevel, BPE, and Unigram.

Quick start::

    from abctokz import Tokenizer
    from abctokz.config.defaults import bpe_multilingual

    config = bpe_multilingual(vocab_size=8000)
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train(["data/corpus.txt"], config)
    tokenizer.save("artifacts/bpe")

    enc = tokenizer.encode("नमस्ते world")
    print(enc.tokens)  # subword pieces
    print(tokenizer.decode(enc.ids))  # reconstructed text
"""

from abctokz.tokenizer import AugenblickTokenizer, Tokenizer
from abctokz.types import ArtifactMetadata, BenchmarkResult, Encoding, SpecialToken
from abctokz.version import __version__

__all__ = [
    "AugenblickTokenizer",
    "Tokenizer",
    "Encoding",
    "ArtifactMetadata",
    "BenchmarkResult",
    "SpecialToken",
    "__version__",
]
