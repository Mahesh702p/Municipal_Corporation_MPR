"""Unit tests for trainers."""

from __future__ import annotations

import pytest

from abctokz.config.schemas import BPETrainerConfig, UnigramTrainerConfig, WordLevelTrainerConfig
from abctokz.trainers.bpe_trainer import BPETrainer
from abctokz.trainers.unigram_trainer import UnigramTrainer
from abctokz.trainers.wordlevel_trainer import WordLevelTrainer


CORPUS = [
    "hello world",
    "hello python",
    "world of python",
    "नमस्ते दुनिया",
    "नमस्ते hello world",
] * 5  # repeat to get enough frequency


class TestWordLevelTrainer:
    def test_trains_basic_vocab(self) -> None:
        cfg = WordLevelTrainerConfig(vocab_size=20, min_frequency=1, special_tokens=["<unk>"])
        trainer = WordLevelTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        vocab = model.get_vocab()
        assert "<unk>" in vocab
        assert "hello" in vocab

    def test_special_tokens_get_lowest_ids(self) -> None:
        cfg = WordLevelTrainerConfig(vocab_size=20, min_frequency=1, special_tokens=["<unk>", "<s>"])
        trainer = WordLevelTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        vocab = model.get_vocab()
        assert vocab["<unk>"] == 0
        assert vocab["<s>"] == 1

    def test_min_frequency_filter(self) -> None:
        cfg = WordLevelTrainerConfig(vocab_size=100, min_frequency=100, special_tokens=["<unk>"])
        trainer = WordLevelTrainer(cfg)
        model = trainer.train(iter(["hello world"]))
        vocab = model.get_vocab()
        # With min_freq=100 and only 1 occurrence, no non-special tokens
        assert "hello" not in vocab

    def test_determinism(self) -> None:
        cfg = WordLevelTrainerConfig(vocab_size=20, min_frequency=1, special_tokens=["<unk>"], seed=42)
        trainer1 = WordLevelTrainer(cfg)
        trainer2 = WordLevelTrainer(cfg)
        model1 = trainer1.train(iter(CORPUS))
        model2 = trainer2.train(iter(CORPUS))
        assert model1.get_vocab() == model2.get_vocab()

    def test_vocab_size_cap(self) -> None:
        cfg = WordLevelTrainerConfig(vocab_size=5, min_frequency=1, special_tokens=["<unk>"])
        trainer = WordLevelTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        assert model.get_vocab_size() <= 5


class TestBPETrainer:
    def test_trains_and_encodes(self) -> None:
        cfg = BPETrainerConfig(vocab_size=50, min_frequency=1, special_tokens=["<unk>"])
        trainer = BPETrainer(cfg)
        model = trainer.train(iter(CORPUS))
        assert model.get_vocab_size() > 0
        result = model.tokenize("hello")
        assert len(result) > 0

    def test_special_tokens_first(self) -> None:
        cfg = BPETrainerConfig(vocab_size=50, min_frequency=1, special_tokens=["<unk>"])
        trainer = BPETrainer(cfg)
        model = trainer.train(iter(CORPUS))
        vocab = model.get_vocab()
        assert "<unk>" in vocab
        assert vocab["<unk>"] == 0

    def test_determinism(self) -> None:
        cfg = BPETrainerConfig(vocab_size=50, min_frequency=1, special_tokens=["<unk>"], seed=42)
        model1 = BPETrainer(cfg).train(iter(CORPUS))
        model2 = BPETrainer(cfg).train(iter(CORPUS))
        assert model1.get_vocab() == model2.get_vocab()

    def test_merges_applied(self) -> None:
        cfg = BPETrainerConfig(vocab_size=100, min_frequency=1, special_tokens=["<unk>"])
        trainer = BPETrainer(cfg)
        # Train on a corpus where "hello" is very frequent
        corpus = ["hello"] * 50 + ["world"] * 50
        model = trainer.train(iter(corpus))
        vocab = model.get_vocab()
        # The vocab must contain at least the individual chars and some merged tokens
        assert "<unk>" in vocab
        # Individual characters should always be present
        assert "h" in vocab
        assert "w" in vocab
        # Some merges must have been learned (vocab > initial chars)
        assert model.get_vocab_size() > 12  # more than individual chars


class TestUnigramTrainer:
    def test_trains_basic(self) -> None:
        cfg = UnigramTrainerConfig(vocab_size=30, special_tokens=["<unk>"])
        trainer = UnigramTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        assert model.get_vocab_size() > 0

    def test_special_tokens_present(self) -> None:
        cfg = UnigramTrainerConfig(vocab_size=30, special_tokens=["<unk>"])
        trainer = UnigramTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        vocab = model.get_vocab()
        assert "<unk>" in vocab

    def test_determinism(self) -> None:
        cfg = UnigramTrainerConfig(vocab_size=30, special_tokens=["<unk>"], seed=42)
        model1 = UnigramTrainer(cfg).train(iter(CORPUS))
        model2 = UnigramTrainer(cfg).train(iter(CORPUS))
        assert model1.get_vocab() == model2.get_vocab()

    def test_encodes_after_training(self) -> None:
        cfg = UnigramTrainerConfig(vocab_size=30, special_tokens=["<unk>"])
        trainer = UnigramTrainer(cfg)
        model = trainer.train(iter(CORPUS))
        result = model.tokenize("hello")
        assert len(result) > 0
