"""Unit tests for normalizers."""

from __future__ import annotations

import unicodedata

import pytest

from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.identity import IdentityNormalizer
from abctokz.normalizers.sequence import SequenceNormalizer
from abctokz.normalizers.unicode_nfkc import NfkcNormalizer
from abctokz.normalizers.whitespace import WhitespaceNormalizer


class TestIdentityNormalizer:
    def test_returns_input_unchanged(self) -> None:
        norm = IdentityNormalizer()
        for text in ["hello", "नमस्ते", "  spaces  ", "", "123"]:
            assert norm.normalize(text) == text

    def test_callable(self) -> None:
        norm = IdentityNormalizer()
        assert norm("hello") == "hello"


class TestNfkcNormalizer:
    def test_fullwidth_ascii(self) -> None:
        norm = NfkcNormalizer()
        assert norm.normalize("ＨＥＬＬＯ") == "HELLO"

    def test_ligatures(self) -> None:
        norm = NfkcNormalizer()
        result = norm.normalize("\ufb01")  # fi ligature
        assert result == "fi"

    def test_strips_zero_width(self) -> None:
        norm = NfkcNormalizer(strip_zero_width=True)
        text = "hello\u200Cworld"
        assert "\u200C" not in norm.normalize(text)

    def test_preserves_zero_width_when_disabled(self) -> None:
        norm = NfkcNormalizer(strip_zero_width=False)
        text = "hello\u200Cworld"
        assert "\u200C" in norm.normalize(text)

    def test_nfc_form(self) -> None:
        norm = NfkcNormalizer()
        text = "cafe\u0301"  # NFD form of "café"
        result = norm.normalize(text)
        # NFKC applies canonical composition
        assert result == unicodedata.normalize("NFKC", text)


class TestWhitespaceNormalizer:
    def test_strips_leading_trailing(self) -> None:
        norm = WhitespaceNormalizer(strip=True)
        assert norm.normalize("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self) -> None:
        norm = WhitespaceNormalizer(collapse=True)
        assert norm.normalize("hello   world") == "hello world"

    def test_no_strip(self) -> None:
        norm = WhitespaceNormalizer(strip=False, collapse=False)
        assert norm.normalize("  hello  ") == "  hello  "

    def test_tabs_collapsed(self) -> None:
        norm = WhitespaceNormalizer(collapse=True)
        assert norm.normalize("hello\t\tworld") == "hello world"

    def test_empty_string(self) -> None:
        norm = WhitespaceNormalizer()
        assert norm.normalize("") == ""


class TestDevanagariNormalizer:
    def test_nfc_applied(self) -> None:
        norm = DevanagariNormalizer(nfc_first=True)
        text = "नमस्ते"
        result = norm.normalize(text)
        assert result == unicodedata.normalize("NFC", text)

    def test_preserves_zwnj_by_default(self) -> None:
        norm = DevanagariNormalizer(strip_zero_width=False)
        text = "क\u200Cख"  # ZWNJ between characters
        assert "\u200C" in norm.normalize(text)

    def test_strips_zwnj_when_enabled(self) -> None:
        norm = DevanagariNormalizer(strip_zero_width=True)
        text = "क\u200Cख"
        assert "\u200C" not in norm.normalize(text)

    def test_ideographic_space_normalized(self) -> None:
        norm = DevanagariNormalizer()
        text = "नमस्ते\u3000world"  # ideographic space
        result = norm.normalize(text)
        assert "\u3000" not in result
        assert " " in result

    def test_mixed_script_preserved(self) -> None:
        norm = DevanagariNormalizer()
        text = "hello नमस्ते"
        result = norm.normalize(text)
        assert "hello" in result
        assert "नमस्ते" in result


class TestSequenceNormalizer:
    def test_empty_sequence(self) -> None:
        norm = SequenceNormalizer([])
        assert norm.normalize("hello") == "hello"

    def test_single_normalizer(self) -> None:
        norm = SequenceNormalizer([WhitespaceNormalizer()])
        assert norm.normalize("  hello  ") == "hello"

    def test_chain_order(self) -> None:
        norm = SequenceNormalizer([
            NfkcNormalizer(),
            WhitespaceNormalizer(strip=True, collapse=True),
        ])
        result = norm.normalize("  ＨＥＬＬＯ  ")
        assert result == "HELLO"

    def test_callable_interface(self) -> None:
        norm = SequenceNormalizer([WhitespaceNormalizer()])
        assert norm("  test  ") == "test"
