"""Integration tests: train → save → load → encode → decode pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
from abctokz.tokenizer import Tokenizer

CORPUS_LINES = [
    "hello world",
    "the quick brown fox",
    "नमस्ते दुनिया",
    "हिन्दी भाषा में परीक्षण",
    "मराठी भाषेत टोकनायझेशन",
    "BPE tokenizer test",
    "hello नमस्ते world दुनिया",
    "tokenization is important",
] * 10  # repeat for frequency


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(CORPUS_LINES), encoding="utf-8")
    return p


@pytest.mark.integration
class TestWordLevelPipeline:
    def test_train_save_load_encode_decode(self, corpus_file: Path, tmp_path: Path) -> None:
        config = wordlevel_multilingual(vocab_size=50)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        artifact_dir = str(tmp_path / "wl_artifact")
        tokenizer.save(artifact_dir)

        loaded = Tokenizer.load(artifact_dir)
        assert loaded.get_vocab_size() > 0

        enc = loaded.encode("hello world")
        assert len(enc.ids) > 0
        assert len(enc.tokens) == len(enc.ids)

        decoded = loaded.decode(enc.ids)
        assert isinstance(decoded, str)

    def test_devanagari_encodes(self, corpus_file: Path, tmp_path: Path) -> None:
        config = wordlevel_multilingual(vocab_size=50)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        enc = tokenizer.encode("नमस्ते दुनिया")
        assert len(enc.ids) > 0

    def test_batch_encode_matches_single(self, corpus_file: Path, tmp_path: Path) -> None:
        config = wordlevel_multilingual(vocab_size=50)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        texts = ["hello world", "नमस्ते"]
        batch = tokenizer.encode_batch(texts)
        singles = [tokenizer.encode(t) for t in texts]
        for b, s in zip(batch, singles):
            assert b.ids == s.ids


@pytest.mark.integration
class TestBPEPipeline:
    def test_train_save_load_encode_decode(self, corpus_file: Path, tmp_path: Path) -> None:
        config = bpe_multilingual(vocab_size=100)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        artifact_dir = str(tmp_path / "bpe_artifact")
        tokenizer.save(artifact_dir)

        loaded = Tokenizer.load(artifact_dir)
        enc = loaded.encode("hello world")
        assert len(enc.ids) > 0

        decoded = loaded.decode(enc.ids)
        assert isinstance(decoded, str)

    def test_mixed_script_encoding(self, corpus_file: Path) -> None:
        config = bpe_multilingual(vocab_size=100)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        enc = tokenizer.encode("hello नमस्ते world")
        assert len(enc.ids) > 0
        decoded = tokenizer.decode(enc.ids)
        assert isinstance(decoded, str)


@pytest.mark.integration
class TestUnigramPipeline:
    def test_train_save_load_encode_decode(self, corpus_file: Path, tmp_path: Path) -> None:
        config = unigram_multilingual(vocab_size=80)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        artifact_dir = str(tmp_path / "unigram_artifact")
        tokenizer.save(artifact_dir)

        loaded = Tokenizer.load(artifact_dir)
        enc = loaded.encode("hello world")
        assert len(enc.ids) > 0

        decoded = loaded.decode(enc.ids)
        assert isinstance(decoded, str)

    def test_devanagari_encodes(self, corpus_file: Path) -> None:
        config = unigram_multilingual(vocab_size=80)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        enc = tokenizer.encode("नमस्ते दुनिया")
        assert len(enc.ids) > 0
