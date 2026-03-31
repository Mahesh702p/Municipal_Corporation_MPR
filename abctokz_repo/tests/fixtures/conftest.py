"""Shared pytest fixtures for abctokz tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from abctokz.vocab.vocab import Vocabulary
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.pieces import PieceTable


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

EN_SENTENCES = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is important for NLP",
    "machine learning models need good tokenizers",
    "english text with punctuation, and numbers 123",
]

HI_SENTENCES = [
    "नमस्ते दुनिया",
    "यह एक परीक्षण वाक्य है",
    "हिन्दी भाषा में टोकनाइजेशन",
    "मशीन लर्निंग मॉडल के लिए टोकनाइज़र",
    "भारत एक विशाल देश है",
]

MR_SENTENCES = [
    "नमस्कार जग",
    "मराठी भाषेत टोकनायझेशन",
    "हे एक चाचणी वाक्य आहे",
]

SD_SENTENCES = [
    "نمستي دنيا",
    "هندي سنڌي",
]

MIXED_SENTENCES = [
    "hello नमस्ते world दुनिया",
    "BPE tokenizer for Hindi हिन्दी",
    "Devanagari script नागरी लिपि",
]

ALL_SENTENCES = EN_SENTENCES + HI_SENTENCES + MR_SENTENCES + MIXED_SENTENCES


# ---------------------------------------------------------------------------
# Vocabulary fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_vocab() -> Vocabulary:
    """A minimal vocabulary for testing."""
    return Vocabulary({"<unk>": 0, "hello": 1, "world": 2, "नमस्ते": 3, "दुनिया": 4})


@pytest.fixture
def simple_merge_table() -> MergeTable:
    """A minimal BPE merge table for testing."""
    rules = [
        (("h", "e"), "he"),
        (("he", "##l"), "hel"),
        (("hel", "##l"), "hell"),
        (("hell", "##o"), "hello"),
    ]
    return MergeTable(rules)


@pytest.fixture
def simple_piece_table() -> PieceTable:
    """A minimal Unigram piece table for testing."""
    return PieceTable([
        ("<unk>", 0.0),
        ("hello", -1.0),
        ("world", -1.5),
        ("नमस्ते", -1.2),
        ("he", -2.0),
        ("llo", -2.5),
    ])


# ---------------------------------------------------------------------------
# Corpus file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Write ALL_SENTENCES to a temporary corpus file."""
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("\n".join(ALL_SENTENCES), encoding="utf-8")
    return corpus_file


@pytest.fixture
def tmp_en_corpus(tmp_path: Path) -> Path:
    """English-only corpus."""
    p = tmp_path / "en_corpus.txt"
    p.write_text("\n".join(EN_SENTENCES * 10), encoding="utf-8")
    return p


@pytest.fixture
def tmp_hi_corpus(tmp_path: Path) -> Path:
    """Hindi-only corpus."""
    p = tmp_path / "hi_corpus.txt"
    p.write_text("\n".join(HI_SENTENCES * 10), encoding="utf-8")
    return p
