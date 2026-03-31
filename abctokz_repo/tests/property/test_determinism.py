"""Property-based tests: determinism and round-trip invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from abctokz.config.defaults import bpe_multilingual, wordlevel_multilingual
from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.whitespace import WhitespaceNormalizer
from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer
from abctokz.vocab.vocab import Vocabulary
from abctokz.models.wordlevel import WordLevelModel


TRAINING_CORPUS = [
    "hello world test",
    "नमस्ते दुनिया",
    "the quick brown fox",
    "machine learning NLP",
] * 20


# ---------------------------------------------------------------------------
# Determinism tests (no Hypothesis required)
# ---------------------------------------------------------------------------

@pytest.mark.property
class TestDeterminism:
    def test_wordlevel_encode_deterministic(self, tmp_path: Path) -> None:
        """Same input always produces same IDs."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("\n".join(TRAINING_CORPUS), encoding="utf-8")

        from abctokz.tokenizer import Tokenizer
        config = wordlevel_multilingual(vocab_size=50)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        text = "hello world"
        enc1 = tokenizer.encode(text)
        enc2 = tokenizer.encode(text)
        assert enc1.ids == enc2.ids
        assert enc1.tokens == enc2.tokens

    def test_bpe_encode_deterministic(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("\n".join(TRAINING_CORPUS), encoding="utf-8")

        from abctokz.tokenizer import Tokenizer
        config = bpe_multilingual(vocab_size=80)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_file)], config)

        text = "hello world"
        enc1 = tokenizer.encode(text)
        enc2 = tokenizer.encode(text)
        assert enc1.ids == enc2.ids

    def test_normalizer_deterministic(self) -> None:
        norm = DevanagariNormalizer()
        text = "नमस्ते world  "
        assert norm.normalize(text) == norm.normalize(text)

    def test_pretokenizer_deterministic(self) -> None:
        pt = WhitespacePreTokenizer()
        text = "hello world foo"
        assert pt.pre_tokenize(text) == pt.pre_tokenize(text)

    def test_vocab_lookup_deterministic(self) -> None:
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        model = WordLevelModel(vocab)
        for _ in range(10):
            assert model.tokenize("hello") == [("hello", 1)]


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.property
class TestRoundTrip:
    def test_wordlevel_round_trip(self, tmp_path: Path) -> None:
        """Encoding then decoding a known word should return the word."""
        vocab = Vocabulary({"<unk>": 0, "hello": 1, "world": 2})
        model = WordLevelModel(vocab)

        tokens_hello = [t for t, _ in model.tokenize("hello")]
        from abctokz.decoders.word_decoder import WordDecoder
        dec = WordDecoder()
        assert dec.decode(tokens_hello) == "hello"

    def test_devanagari_normalizer_round_trip(self) -> None:
        """NFC normalization of already-NFC text should be idempotent."""
        import unicodedata
        norm = DevanagariNormalizer(nfc_first=True)
        texts = ["नमस्ते", "मराठी", "hello world", "सिन्धी"]
        for text in texts:
            normalized = norm.normalize(text)
            # Applying twice should give the same result
            assert norm.normalize(normalized) == normalized


# ---------------------------------------------------------------------------
# Hypothesis property tests (skip if not installed)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    @pytest.mark.property
    @given(st.text(alphabet=st.characters(whitelist_categories=("L", "Z")), min_size=0, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_whitespace_normalizer_idempotent(text: str) -> None:
        """Applying whitespace normalization twice yields the same result."""
        norm = WhitespaceNormalizer(strip=True, collapse=True)
        once = norm.normalize(text)
        twice = norm.normalize(once)
        assert once == twice

    @pytest.mark.property
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=30))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_whitespace_pretokenizer_no_empty_tokens(text: str) -> None:
        """WhitespacePreTokenizer must never emit empty strings."""
        pt = WhitespacePreTokenizer()
        tokens = pt.pre_tokenize(text)
        assert all(t for t in tokens)
