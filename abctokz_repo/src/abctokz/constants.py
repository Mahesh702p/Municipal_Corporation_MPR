# Augenblick — abctokz
"""Global constants used throughout abctokz."""

from __future__ import annotations

# Special token defaults
UNK_TOKEN: str = "<unk>"
BOS_TOKEN: str = "<s>"
EOS_TOKEN: str = "</s>"
PAD_TOKEN: str = "<pad>"
MASK_TOKEN: str = "<mask>"
SEP_TOKEN: str = "<sep>"
CLS_TOKEN: str = "<cls>"

# Reserved IDs
UNK_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = 3

# Vocabulary defaults
DEFAULT_VOCAB_SIZE: int = 8_000
MIN_FREQUENCY: int = 2
MAX_TOKEN_LENGTH: int = 128

# BPE defaults
BPE_DEFAULT_VOCAB_SIZE: int = 8_000
BPE_DEFAULT_MIN_FREQUENCY: int = 2
BPE_CONTINUATION_PREFIX: str = "##"

# Unigram defaults
UNIGRAM_DEFAULT_VOCAB_SIZE: int = 8_000
UNIGRAM_SHRINKING_FACTOR: float = 0.75
UNIGRAM_NUM_SUB_ITERATIONS: int = 2
UNIGRAM_CHAR_COVERAGE: float = 0.9995

# Devanagari Unicode ranges
DEVANAGARI_START: int = 0x0900
DEVANAGARI_END: int = 0x097F
DEVANAGARI_EXTENDED_START: int = 0xA8E0
DEVANAGARI_EXTENDED_END: int = 0xA8FF
VEDIC_EXTENSIONS_START: int = 0x1CD0
VEDIC_EXTENSIONS_END: int = 0x1CFF

# Artifact filenames
MANIFEST_FILENAME: str = "manifest.json"
CONFIG_FILENAME: str = "config.json"
VOCAB_FILENAME: str = "vocab.json"
MERGES_FILENAME: str = "merges.txt"
PIECES_FILENAME: str = "pieces.json"
SPECIAL_TOKENS_FILENAME: str = "special_tokens.json"

# Benchmark defaults
BENCHMARK_SAMPLE_SIZE: int = 1_000
BENCHMARK_WARMUP_RUNS: int = 3
BENCHMARK_TIMED_RUNS: int = 10

# Logging
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
