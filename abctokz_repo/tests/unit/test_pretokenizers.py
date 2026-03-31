"""Unit tests for pre-tokenizers."""

from __future__ import annotations

import pytest

from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer
from abctokz.pretokenizers.punctuation import PunctuationPreTokenizer
from abctokz.pretokenizers.regex import RegexPreTokenizer
from abctokz.pretokenizers.sequence import SequencePreTokenizer
from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer


class TestWhitespacePreTokenizer:
    def test_basic_split(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("hello world") == ["hello", "world"]

    def test_multiple_spaces(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("a  b   c") == ["a", "b", "c"]

    def test_empty_string(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("") == []

    def test_leading_trailing_spaces(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("  hello  ") == ["hello"]

    def test_callable(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt("hello world") == ["hello", "world"]

    def test_devanagari(self) -> None:
        pt = WhitespacePreTokenizer()
        assert pt.pre_tokenize("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]


class TestPunctuationPreTokenizer:
    def test_isolated_default(self) -> None:
        pt = PunctuationPreTokenizer(behavior="isolated")
        result = pt.pre_tokenize("hello, world!")
        assert "," in result
        assert "!" in result
        assert "hello" in result
        assert "world" in result

    def test_no_punct(self) -> None:
        pt = PunctuationPreTokenizer()
        assert pt.pre_tokenize("hello world") == ["hello", "world"]

    def test_invalid_behavior(self) -> None:
        with pytest.raises(ValueError):
            PunctuationPreTokenizer(behavior="invalid")  # type: ignore[arg-type]

    def test_empty(self) -> None:
        pt = PunctuationPreTokenizer()
        assert pt.pre_tokenize("") == []


class TestRegexPreTokenizer:
    def test_split_on_whitespace(self) -> None:
        pt = RegexPreTokenizer(r"\s+")
        assert pt.pre_tokenize("hello world") == ["hello", "world"]

    def test_invert_keeps_matches(self) -> None:
        pt = RegexPreTokenizer(r"\w+", invert=True)
        result = pt.pre_tokenize("hello, world!")
        assert "hello" in result
        assert "world" in result
        assert "," not in result

    def test_empty_input(self) -> None:
        pt = RegexPreTokenizer(r"\s+")
        assert pt.pre_tokenize("") == []


class TestDevanagariAwarePreTokenizer:
    def test_basic_devanagari_split(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("नमस्ते world")
        assert "नमस्ते" in result
        assert "world" in result

    def test_script_boundary_split(self) -> None:
        pt = DevanagariAwarePreTokenizer(split_on_script_boundary=True)
        result = pt.pre_tokenize("नमस्तेworld")
        assert len(result) == 2
        assert "नमस्ते" in result
        assert "world" in result

    def test_no_script_boundary_split(self) -> None:
        pt = DevanagariAwarePreTokenizer(split_on_script_boundary=False)
        result = pt.pre_tokenize("नमस्ते world")
        # Still splits on whitespace
        assert len(result) == 2

    def test_pure_devanagari(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("नमस्ते दुनिया")
        assert result == ["नमस्ते", "दुनिया"]

    def test_matra_preserved(self) -> None:
        """Combining marks (matras) must stay with their base char."""
        pt = DevanagariAwarePreTokenizer()
        # 'की' = क + ी (matra) — must not be split
        result = pt.pre_tokenize("की")
        assert len(result) == 1
        assert result[0] == "की"


class TestSequencePreTokenizer:
    def test_chain_whitespace_and_punct(self) -> None:
        pt = SequencePreTokenizer([
            WhitespacePreTokenizer(),
            PunctuationPreTokenizer(behavior="isolated"),
        ])
        result = pt.pre_tokenize("hello, world!")
        assert "hello" in result
        assert "," in result
        assert "world" in result
        assert "!" in result

    def test_empty_chain(self) -> None:
        pt = SequencePreTokenizer([])
        assert pt.pre_tokenize("hello") == ["hello"]

    def test_further_splits_each_token(self) -> None:
        pt = SequencePreTokenizer([
            WhitespacePreTokenizer(),
            RegexPreTokenizer(r"\d+", invert=True),
        ])
        # "hello 123 world" -> whitespace -> ["hello", "123", "world"]
        # regex invert (keep \d+) on "123" -> ["123"]; on "hello" -> []
        result = pt.pre_tokenize("hello 123 world")
        assert "123" in result
