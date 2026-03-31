"""Unit tests for tokenization models."""

from __future__ import annotations

import pytest

from abctokz.models.bpe import BPEModel
from abctokz.models.unigram import UnigramModel
from abctokz.models.wordlevel import WordLevelModel
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.pieces import PieceTable
from abctokz.vocab.vocab import Vocabulary


class TestWordLevelModel:
    def test_known_token(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        model = WordLevelModel(vocab)
        result = model.tokenize("hello")
        assert result == [("hello", 1)]

    def test_unknown_token_returns_unk(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1})
        model = WordLevelModel(vocab)
        result = model.tokenize("xyz")
        assert result[0][0] == "<unk>"
        assert result[0][1] == 0

    def test_devanagari_token(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "नमस्ते": 1})
        model = WordLevelModel(vocab)
        result = model.tokenize("नमस्ते")
        assert result == [("नमस्ते", 1)]

    def test_get_vocab(self) -> None:
        vocab_dict = {"<unk>": 0, "a": 1}
        vocab = Vocabulary(vocab_dict)
        model = WordLevelModel(vocab)
        assert model.get_vocab() == vocab_dict

    def test_get_vocab_size(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "a": 1, "b": 2})
        model = WordLevelModel(vocab)
        assert model.get_vocab_size() == 3

    def test_save_load(self, tmp_path: object) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            vocab = Vocabulary({"<unk>": 0, "hello": 1})
            model = WordLevelModel(vocab)
            model.save(d)
            loaded = WordLevelModel.load(d)
            assert loaded.get_vocab() == model.get_vocab()


class TestBPEModel:
    def _make_model(self) -> BPEModel:
        vocab = Vocabulary({
            "<unk>": 0, "h": 1, "##e": 2, "he": 3,
            "##l": 4, "##o": 5, "hel": 6, "hell": 7, "hello": 8,
            "w": 9, "wo": 10, "wor": 11, "worl": 12, "world": 13,
        })
        merges = MergeTable([
            (("h", "##e"), "he"),
            (("he", "##l"), "hel"),
            (("hel", "##l"), "hell"),
            (("hell", "##o"), "hello"),
            (("w", "##o"), "wo"),
            (("wo", "##r"), "wor"),
            (("wor", "##l"), "worl"),
            (("worl", "##d"), "world"),
        ])
        return BPEModel(vocab, merges)

    def test_known_word_merges_fully(self) -> None:
        model = self._make_model()
        result = model.tokenize("hello")
        # Should fully merge to "hello"
        tokens = [t for t, _ in result]
        assert "hello" in tokens

    def test_empty_sequence(self) -> None:
        model = self._make_model()
        assert model.tokenize("") == []

    def test_get_vocab_size(self) -> None:
        model = self._make_model()
        assert model.get_vocab_size() > 0

    def test_save_load(self, tmp_path: object) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            model = self._make_model()
            model.save(d)
            loaded = BPEModel.load(d)
            result_orig = model.tokenize("hello")
            result_loaded = loaded.tokenize("hello")
            assert result_orig == result_loaded


class TestUnigramModel:
    def _make_model(self) -> UnigramModel:
        pieces = PieceTable([
            ("<unk>", 0.0),
            ("hello", -1.0),
            ("world", -1.5),
            ("नमस्ते", -1.2),
            ("he", -2.0),
            ("ll", -2.5),
            ("o", -3.0),
            ("w", -3.0),
            ("r", -3.0),
            ("l", -3.0),
            ("d", -3.0),
        ])
        return UnigramModel(pieces)

    def test_known_piece(self) -> None:
        model = self._make_model()
        result = model.tokenize("hello")
        tokens = [t for t, _ in result]
        assert "hello" in tokens

    def test_devanagari_word(self) -> None:
        model = self._make_model()
        result = model.tokenize("नमस्ते")
        tokens = [t for t, _ in result]
        assert "नमस्ते" in tokens

    def test_empty_sequence(self) -> None:
        model = self._make_model()
        assert model.tokenize("") == []

    def test_save_load(self, tmp_path: object) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            model = self._make_model()
            model.save(d)
            loaded = UnigramModel.load(d)
            result_orig = model.tokenize("hello")
            result_loaded = loaded.tokenize("hello")
            assert result_orig == result_loaded
