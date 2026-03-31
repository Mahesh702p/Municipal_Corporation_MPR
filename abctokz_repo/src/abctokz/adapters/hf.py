# Augenblick — abctokz
"""Hugging Face tokenizers adapter.

Wraps a HF ``tokenizers.Tokenizer`` object behind the same encode/decode
interface as :class:`~abctokz.tokenizer.Tokenizer`, allowing direct comparison
in benchmarks.
"""

from __future__ import annotations

from typing import Any

from abctokz.exceptions import AdapterError
from abctokz.types import BenchmarkResult, Encoding


class HFTokenizerAdapter:
    """Wraps a Hugging Face tokenizer for use in abctokz benchmarks.

    Requires ``tokenizers`` (the Rust-backed library) to be installed.

    Args:
        model_name_or_path: HF model name (e.g. ``"bert-base-multilingual-cased"``)
            or path to a saved tokenizer directory.

    Example::

        from abctokz.adapters.hf import HFTokenizerAdapter
        adapter = HFTokenizerAdapter("bert-base-multilingual-cased")
        enc = adapter.encode("hello world")
        print(enc.ids)
    """

    def __init__(self, model_name_or_path: str) -> None:
        try:
            from tokenizers import Tokenizer as _HFTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise AdapterError(
                "The 'tokenizers' package is required for HFTokenizerAdapter. "
                "Install it with: pip install tokenizers"
            ) from exc

        try:
            # Try loading as a pretrained tokenizer via transformers
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
            _auto = AutoTokenizer.from_pretrained(model_name_or_path)
            self._tokenizer = _auto
            self._name = model_name_or_path
            self._use_transformers = True
        except Exception:
            try:
                self._tokenizer = _HFTokenizer.from_file(model_name_or_path)
                self._name = model_name_or_path
                self._use_transformers = False
            except Exception as exc2:
                raise AdapterError(f"Could not load HF tokenizer from {model_name_or_path!r}: {exc2}") from exc2

    @property
    def name(self) -> str:
        """Tokenizer identifier."""
        return self._name

    def encode(self, text: str) -> Encoding:
        """Encode *text* using the HF tokenizer.

        Args:
            text: Input string.

        Returns:
            :class:`~abctokz.types.Encoding` with ids and tokens.
        """
        if self._use_transformers:
            out = self._tokenizer(text, return_attention_mask=True)
            ids = out["input_ids"]
            tokens = self._tokenizer.convert_ids_to_tokens(ids) or []
            attn = out.get("attention_mask", [1] * len(ids))
        else:
            out = self._tokenizer.encode(text)
            ids = out.ids
            tokens = out.tokens
            attn = out.attention_mask if hasattr(out, "attention_mask") else [1] * len(ids)

        return Encoding(
            ids=ids,
            tokens=tokens if tokens else [""] * len(ids),
            attention_mask=list(attn),
        )

    def encode_batch(self, texts: list[str]) -> list[Encoding]:
        """Encode a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of encodings.
        """
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode *ids* back to a string.

        Args:
            ids: Token IDs.
            skip_special_tokens: Skip special tokens in output.

        Returns:
            Decoded string.
        """
        if self._use_transformers:
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return self._tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self._use_transformers:
            return len(self._tokenizer)
        return self._tokenizer.get_vocab_size()
