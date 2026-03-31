"""Golden tests: verify fixed tokenization outputs for specific inputs.

These tests use a pre-built in-memory tokenizer to verify correctness of
the encoding logic itself, not training.
"""

from __future__ import annotations

import pytest

from abctokz.models.wordlevel import WordLevelModel
from abctokz.models.bpe import BPEModel
from abctokz.vocab.vocab import Vocabulary
from abctokz.vocab.merges import MergeTable
from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer


# ---------------------------------------------------------------------------
# English golden tests
# ---------------------------------------------------------------------------

@pytest.mark.golden
class TestEnglishGolden:
    def test_wordlevel_known_words(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2, "the": 3, "quick": 4})
        model = WordLevelModel(vocab)
        assert model.tokenize("hello") == [("hello", 1)]
        assert model.tokenize("world") == [("world", 2)]

    def test_wordlevel_unknown_word(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1})
        model = WordLevelModel(vocab)
        result = model.tokenize("xyz")
        assert result[0][0] == "<unk>"
        assert result[0][1] == 0

    def test_bpe_fully_merges_common_word(self) -> None:
        """A word that appears as a full-vocab entry should encode to a single token."""
        vocab = Vocabulary({
            "<unk>": 0, "h": 1, "##e": 2, "##l": 3, "##o": 4,
            "he": 5, "hel": 6, "hell": 7, "hello": 8,
        })
        merges = MergeTable([
            (("h", "##e"), "he"),
            (("he", "##l"), "hel"),
            (("hel", "##l"), "hell"),
            (("hell", "##o"), "hello"),
        ])
        model = BPEModel(vocab, merges)
        result = model.tokenize("hello")
        assert result == [("hello", 8)]


# ---------------------------------------------------------------------------
# Hindi golden tests
# ---------------------------------------------------------------------------

@pytest.mark.golden
class TestHindiGolden:
    def test_devanagari_normalizer_nfc(self) -> None:
        import unicodedata
        norm = DevanagariNormalizer(nfc_first=True)
        text = "नमस्ते"
        assert norm.normalize(text) == unicodedata.normalize("NFC", text)

    def test_devanagari_pretokenizer_splits_on_whitespace(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("नमस्ते दुनिया")
        assert result == ["नमस्ते", "दुनिया"]

    def test_wordlevel_hindi_token(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "नमस्ते": 1, "दुनिया": 2})
        model = WordLevelModel(vocab)
        assert model.tokenize("नमस्ते") == [("नमस्ते", 1)]
        assert model.tokenize("दुनिया") == [("दुनिया", 2)]

    def test_matra_preserved_in_pretokenizer(self) -> None:
        """Matras (vowel markers) must not be split from their base consonants."""
        pt = DevanagariAwarePreTokenizer()
        # 'की' = क (ka) + ी (i matra)
        result = pt.pre_tokenize("की बात")
        assert "की" in result
        assert "बात" in result


# ---------------------------------------------------------------------------
# Marathi golden tests
# ---------------------------------------------------------------------------

@pytest.mark.golden
class TestMarathiGolden:
    def test_marathi_words_preserved(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("मराठी भाषा")
        assert "मराठी" in result
        assert "भाषा" in result

    def test_marathi_wordlevel(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "मराठी": 1, "भाषा": 2})
        model = WordLevelModel(vocab)
        assert model.tokenize("मराठी") == [("मराठी", 1)]


# ---------------------------------------------------------------------------
# Devanagari Sindhi golden tests
# ---------------------------------------------------------------------------

@pytest.mark.golden
class TestSindhiGolden:
    def test_sindhi_devanagari_preserved(self) -> None:
        """Sindhi written in Devanagari is phonetically rich; tokens must be intact."""
        pt = DevanagariAwarePreTokenizer()
        # Sindhi Devanagari example
        result = pt.pre_tokenize("सिन्धी भाषा")
        assert "सिन्धी" in result
        assert "भाषा" in result


# ---------------------------------------------------------------------------
# Mixed script golden tests
# ---------------------------------------------------------------------------

@pytest.mark.golden
class TestMixedScriptGolden:
    def test_script_boundary_detection(self) -> None:
        pt = DevanagariAwarePreTokenizer(split_on_script_boundary=True)
        result = pt.pre_tokenize("helloनमस्ते")
        assert "hello" in result
        assert "नमस्ते" in result

    def test_mixed_sentence_splits_correctly(self) -> None:
        pt = DevanagariAwarePreTokenizer()
        result = pt.pre_tokenize("BPE tokenizer नमस्ते world")
        assert "BPE" in result
        assert "नमस्ते" in result
        assert "world" in result
