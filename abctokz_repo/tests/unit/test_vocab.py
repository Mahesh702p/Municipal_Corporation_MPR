"""Unit tests for vocabulary components."""

from __future__ import annotations

import pytest

from abctokz.exceptions import UnknownTokenError, VocabError
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.pieces import PieceTable
from abctokz.vocab.vocab import Vocabulary


class TestVocabulary:
    def test_basic_lookup(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        assert vocab.token_to_id("hello") == 1
        assert vocab.id_to_token(2) == "world"

    def test_unk_fallback(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1}, unk_token="<unk>")
        assert vocab.token_to_id("missing") == 0

    def test_unknown_token_raises_without_unk(self) -> None:
        vocab = Vocabulary({"hello": 0}, unk_token=None)
        with pytest.raises(UnknownTokenError):
            vocab.token_to_id("missing")

    def test_unknown_id_raises(self) -> None:
        vocab = Vocabulary({"hello": 0})
        with pytest.raises(VocabError):
            vocab.id_to_token(999)

    def test_duplicate_ids_raises(self) -> None:
        with pytest.raises(VocabError):
            Vocabulary({"a": 0, "b": 0})

    def test_contains(self) -> None:
        vocab = Vocabulary({"hello": 0})
        assert "hello" in vocab
        assert "world" not in vocab

    def test_len(self) -> None:
        vocab = Vocabulary({"a": 0, "b": 1, "c": 2})
        assert len(vocab) == 3

    def test_to_dict_round_trip(self) -> None:
        data = {"<unk>": 0, "hello": 1, "world": 2}
        vocab = Vocabulary(data)
        assert vocab.to_dict() == data

    def test_from_dict(self) -> None:
        vocab = Vocabulary.from_dict({"hello": 0, "world": 1}, unk_token=None)
        assert vocab.token_to_id("hello") == 0

    def test_size_property(self) -> None:
        vocab = Vocabulary({"a": 0, "b": 1})
        assert vocab.size == 2


class TestMergeTable:
    def test_rank_lookup(self) -> None:
        table = MergeTable([(("h", "e"), "he"), (("he", "##l"), "hel")])
        assert table.get_rank(("h", "e")) == 0
        assert table.get_rank(("he", "##l")) == 1
        assert table.get_rank(("x", "y")) is None

    def test_merge_result(self) -> None:
        table = MergeTable([(("h", "e"), "he")])
        assert table.merge_result(("h", "e")) == "he"
        assert table.merge_result(("x", "y")) is None

    def test_contains(self) -> None:
        table = MergeTable([(("h", "e"), "he")])
        assert ("h", "e") in table
        assert ("x", "y") not in table

    def test_len(self) -> None:
        table = MergeTable([(("a", "b"), "ab"), (("ab", "c"), "abc")])
        assert len(table) == 2

    def test_text_round_trip(self) -> None:
        rules = [(("h", "e"), "he"), (("he", "##l"), "hel")]
        table = MergeTable(rules)
        text = table.to_text()
        loaded = MergeTable.from_text(text)
        assert loaded.rules == rules

    def test_list_round_trip(self) -> None:
        rules = [(("a", "b"), "ab")]
        table = MergeTable(rules)
        loaded = MergeTable.from_list(table.to_list())
        assert loaded.rules == rules


class TestPieceTable:
    def test_score_lookup(self) -> None:
        table = PieceTable([("<unk>", 0.0), ("hello", -1.0)])
        assert table.score("hello") == -1.0
        assert table.score("missing") is None

    def test_piece_to_id(self) -> None:
        table = PieceTable([("a", 0.0), ("b", -1.0)])
        assert table.piece_to_id("a") == 0
        assert table.piece_to_id("b") == 1
        assert table.piece_to_id("c") is None

    def test_id_to_piece(self) -> None:
        table = PieceTable([("a", 0.0), ("b", -1.0)])
        assert table.id_to_piece(0) == "a"
        assert table.id_to_piece(1) == "b"
        assert table.id_to_piece(99) is None

    def test_list_round_trip(self) -> None:
        pieces = [("<unk>", 0.0), ("hello", -1.5), ("world", -2.0)]
        table = PieceTable(pieces)
        loaded = PieceTable.from_list(table.to_list())
        assert loaded.pieces == pieces

    def test_len(self) -> None:
        table = PieceTable([("a", 0.0), ("b", -1.0)])
        assert len(table) == 2
