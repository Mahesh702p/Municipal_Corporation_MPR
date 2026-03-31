# Augenblick — abctokz
"""Data utilities subpackage for abctokz."""

from abctokz.data.corpus import iter_corpus, iter_lines, load_corpus
from abctokz.data.manifest import CorpusEntry, DataManifest
from abctokz.data.sampling import sample_lines, stratified_sample
from abctokz.data.streaming import batched, stream_shards

__all__ = [
    "iter_corpus",
    "iter_lines",
    "load_corpus",
    "CorpusEntry",
    "DataManifest",
    "sample_lines",
    "stratified_sample",
    "batched",
    "stream_shards",
]

