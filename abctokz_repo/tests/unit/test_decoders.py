"""Unit tests for decoders."""

from __future__ import annotations

from abctokz.decoders.subword_decoder import SubwordDecoder
from abctokz.decoders.word_decoder import WordDecoder


class TestWordDecoder:
    def test_basic_join(self) -> None:
        dec = WordDecoder()
        assert dec.decode(["hello", "world"]) == "hello world"

    def test_empty_list(self) -> None:
        dec = WordDecoder()
        assert dec.decode([]) == ""

    def test_skip_special_tokens(self) -> None:
        dec = WordDecoder(skip_special_tokens=True)
        result = dec.decode(["<s>", "hello", "world", "</s>"])
        assert "<s>" not in result
        assert "</s>" not in result
        assert "hello" in result

    def test_custom_separator(self) -> None:
        dec = WordDecoder(separator="-")
        assert dec.decode(["a", "b", "c"]) == "a-b-c"

    def test_callable(self) -> None:
        dec = WordDecoder()
        assert dec(["hello"]) == "hello"


class TestSubwordDecoder:
    def test_bpe_continuation_prefix(self) -> None:
        dec = SubwordDecoder(continuation_prefix="##")
        result = dec.decode(["hello", "##world"])
        assert result == "helloworld"

    def test_bpe_word_boundary(self) -> None:
        dec = SubwordDecoder(continuation_prefix="##")
        result = dec.decode(["good", "bye"])
        assert result == "good bye"

    def test_space_prefix_mode(self) -> None:
        dec = SubwordDecoder(space_prefix="▁")
        result = dec.decode(["▁hello", "▁world"])
        assert result == "hello world"

    def test_space_prefix_no_leading_space(self) -> None:
        dec = SubwordDecoder(space_prefix="▁")
        result = dec.decode(["▁hello"])
        assert not result.startswith(" ")

    def test_empty_tokens(self) -> None:
        dec = SubwordDecoder()
        assert dec.decode([]) == ""

    def test_skip_special_tokens(self) -> None:
        dec = SubwordDecoder(skip_special_tokens=True)
        result = dec.decode(["<unk>", "hello", "<pad>"])
        assert "<unk>" not in result
        assert "hello" in result

    def test_mixed_bpe_output(self) -> None:
        dec = SubwordDecoder(continuation_prefix="##")
        result = dec.decode(["un", "##common"])
        assert result == "uncommon"
