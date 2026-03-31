# Augenblick — abctokz
"""SentencePiece adapter.

Wraps a ``sentencepiece.SentencePieceProcessor`` behind the abctokz interface
for use as a benchmark baseline.
"""

from __future__ import annotations

from abctokz.exceptions import AdapterError
from abctokz.types import Encoding


class SentencePieceAdapter:
    """Wraps a SentencePiece model for use in abctokz benchmarks.

    Requires the ``sentencepiece`` package.

    Args:
        model_path: Path to a ``.model`` file trained by SentencePiece.

    Example::

        from abctokz.adapters.sentencepiece import SentencePieceAdapter
        adapter = SentencePieceAdapter("models/hi.model")
        enc = adapter.encode("नमस्ते world")
        print(enc.tokens)
    """

    def __init__(self, model_path: str) -> None:
        try:
            import sentencepiece as spm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise AdapterError(
                "The 'sentencepiece' package is required for SentencePieceAdapter. "
                "Install it with: pip install sentencepiece"
            ) from exc

        self._sp = spm.SentencePieceProcessor()
        if not self._sp.Load(model_path):
            raise AdapterError(f"Failed to load SentencePiece model from {model_path!r}")
        self._name = model_path

    @property
    def name(self) -> str:
        """Model identifier."""
        return self._name

    def encode(self, text: str) -> Encoding:
        """Encode *text*.

        Args:
            text: Input string.

        Returns:
            :class:`~abctokz.types.Encoding`.
        """
        ids: list[int] = self._sp.EncodeAsIds(text)
        pieces: list[str] = self._sp.EncodeAsPieces(text)
        return Encoding(
            ids=ids,
            tokens=pieces,
            attention_mask=[1] * len(ids),
        )

    def encode_batch(self, texts: list[str]) -> list[Encoding]:
        """Encode a batch of texts.

        Args:
            texts: Input strings.

        Returns:
            List of encodings.
        """
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode *ids* to a string.

        Args:
            ids: Token IDs.
            skip_special_tokens: Unused (SentencePiece handles this internally).

        Returns:
            Decoded string.
        """
        return self._sp.DecodeIds(ids)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._sp.GetPieceSize()
