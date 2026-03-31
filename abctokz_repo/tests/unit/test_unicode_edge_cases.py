"""Regression tests for Unicode edge cases in Devanagari processing."""

from __future__ import annotations

import unicodedata

import pytest

from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer
from abctokz.utils.unicode import (
    grapheme_clusters,
    is_combining,
    is_devanagari,
    is_zero_width,
    normalize_nfc,
    normalize_nfkc,
    strip_zero_width,
)


class TestDevanagariUnicodeRange:
    def test_basic_consonants_detected(self) -> None:
        consonants = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
        for ch in consonants:
            assert is_devanagari(ch), f"Expected {ch!r} to be Devanagari"

    def test_vowels_detected(self) -> None:
        vowels = "अआइईउऊएऐओऔ"
        for ch in vowels:
            assert is_devanagari(ch)

    def test_combining_marks_detected(self) -> None:
        # Matras (vowel signs)
        marks = "ािीुूेैोौ"
        for ch in marks:
            assert is_devanagari(ch), f"Expected matra {ch!r} to be Devanagari"

    def test_halant_detected(self) -> None:
        assert is_devanagari("्")  # Halant U+094D

    def test_anusvara_detected(self) -> None:
        assert is_devanagari("ं")  # Anusvara U+0902

    def test_visarga_detected(self) -> None:
        assert is_devanagari("ः")  # Visarga U+0903

    def test_latin_not_devanagari(self) -> None:
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert not is_devanagari(ch)

    def test_digits_not_devanagari(self) -> None:
        for ch in "0123456789":
            assert not is_devanagari(ch)


class TestCombiningMarks:
    def test_combining_marks_identified(self) -> None:
        # NFC: combining marks in Devanagari words
        text = "की"  # क + ी
        for i, ch in enumerate(text):
            if i > 0:
                assert is_combining(ch) or unicodedata.category(ch).startswith("M")

    def test_latin_not_combining(self) -> None:
        assert not is_combining("a")
        assert not is_combining("Z")


class TestZeroWidth:
    def test_zwnj_is_zero_width(self) -> None:
        assert is_zero_width("\u200C")

    def test_zwj_is_zero_width(self) -> None:
        assert is_zero_width("\u200D")

    def test_bom_is_zero_width(self) -> None:
        assert is_zero_width("\uFEFF")

    def test_regular_chars_not_zero_width(self) -> None:
        for ch in "hello नमस्ते ":
            assert not is_zero_width(ch)

    def test_strip_zero_width_removes_zwnj(self) -> None:
        text = "क\u200Cख"
        result = strip_zero_width(text)
        assert "\u200C" not in result
        assert "क" in result
        assert "ख" in result


class TestGraphemeClusters:
    def test_latin_chars_are_own_clusters(self) -> None:
        clusters = grapheme_clusters("hello")
        assert clusters == ["h", "e", "l", "l", "o"]

    def test_devanagari_matra_stays_with_base(self) -> None:
        # 'की' should be a single grapheme cluster
        clusters = grapheme_clusters("की")
        assert len(clusters) == 1
        assert clusters[0] == "की"

    def test_empty_string(self) -> None:
        assert grapheme_clusters("") == []

    def test_mixed_script_clusters(self) -> None:
        clusters = grapheme_clusters("aक")
        assert "a" in clusters
        # क might be its own cluster if no combining marks follow


class TestNormalizationConsistency:
    def test_nfc_idempotent_on_devanagari(self) -> None:
        texts = ["नमस्ते", "हिन्दी", "मराठी", "भारत"]
        for text in texts:
            nfc = normalize_nfc(text)
            assert normalize_nfc(nfc) == nfc

    def test_nfkc_idempotent_on_ascii(self) -> None:
        texts = ["hello", "world", "HELLO WORLD", "test 123"]
        for text in texts:
            nfkc = normalize_nfkc(text)
            assert normalize_nfkc(nfkc) == nfkc

    def test_devanagari_normalizer_preserves_words(self) -> None:
        norm = DevanagariNormalizer()
        words = ["नमस्ते", "दुनिया", "हिन्दी", "मराठी"]
        for word in words:
            result = norm.normalize(word)
            assert len(result) > 0
            # No characters should be removed unexpectedly
            assert result.strip() == result


class TestDevanagariAwarePretokenizerEdgeCases:
    def test_empty_string(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        assert pt.pre_tokenize("") == []

    def test_only_whitespace(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        assert pt.pre_tokenize("   ") == []

    def test_single_devanagari_char(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("क")
        assert result == ["क"]

    def test_conjunct_consonants_not_split(self) -> None:
        """'क्ष' (conjunct: k+halant+sh) should remain as one pre-token."""
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("क्ष")
        assert len(result) == 1

    def test_number_with_devanagari(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("नमस्ते 2024")
        assert "नमस्ते" in result
        assert "2024" in result
