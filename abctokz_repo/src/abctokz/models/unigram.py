# Augenblick — abctokz
"""Unigram language model tokenizer."""

from __future__ import annotations

import math
from pathlib import Path

from abctokz.constants import UNK_TOKEN
from abctokz.models.base import Model
from abctokz.vocab.pieces import PieceTable
from abctokz.vocab.serialization import load_pieces, save_pieces


class UnigramModel(Model):
    """Unigram language model tokenizer.

    Tokenizes a pre-token by finding the segmentation that maximizes the
    sum of log-probabilities of the pieces (Viterbi decoding).

    Unknown characters are handled by emitting the ``<unk>`` piece with
    score 0.0 for that character position.

    Args:
        pieces: :class:`~abctokz.vocab.pieces.PieceTable` with piece scores.
        unk_token: Unknown token string.
        unk_id: Vocabulary ID for the unknown token.

    Example::

        from abctokz.vocab.pieces import PieceTable
        table = PieceTable([("<unk>", 0.0), ("▁he", -1.0), ("▁hello", -2.0), ("llo", -1.5)])
        model = UnigramModel(table)
        result = model.tokenize("▁hello")
        # -> [("▁hello", 2)] (single piece preferred if score better)
    """

    NEG_INF: float = -1e9

    def __init__(
        self,
        pieces: PieceTable,
        unk_token: str = UNK_TOKEN,
        unk_id: int = 0,
    ) -> None:
        self._pieces = pieces
        self._unk_token = unk_token
        self._unk_id = unk_id
        # Build max piece length for pruning the DP
        self._max_len = max((len(p) for p, _ in pieces.pieces if p != unk_token), default=1)

    def tokenize(self, sequence: str) -> list[tuple[str, int]]:
        """Tokenize *sequence* using Viterbi Unigram decoding.

        Args:
            sequence: A single pre-token string.

        Returns:
            List of ``(piece_string, piece_id)`` pairs representing the
            best segmentation.
        """
        if not sequence:
            return []
        return self._viterbi(sequence)

    def _viterbi(self, text: str) -> list[tuple[str, int]]:
        """Run the Viterbi algorithm for Unigram segmentation.

        Args:
            text: Input string to segment.

        Returns:
            Best segmentation as ``(piece, id)`` pairs.
        """
        n = len(text)
        # best[i] = (best_score, best_start) ending at position i
        best_score = [self.NEG_INF] * (n + 1)
        best_start = [-1] * (n + 1)
        best_score[0] = 0.0

        for end in range(1, n + 1):
            for start in range(max(0, end - self._max_len), end):
                if best_score[start] == self.NEG_INF:
                    continue
                piece = text[start:end]
                piece_score = self._pieces.score(piece)
                if piece_score is None:
                    # Unknown character: only allow single-char fallback
                    if end - start == 1:
                        piece_score = self.NEG_INF / 2  # heavy penalty
                    else:
                        continue
                candidate = best_score[start] + piece_score
                if candidate > best_score[end]:
                    best_score[end] = candidate
                    best_start[end] = start

        # Backtrack
        pieces: list[tuple[str, int]] = []
        pos = n
        while pos > 0:
            start = best_start[pos]
            if start < 0:
                # Fallback: emit single unknown char
                char = text[pos - 1]
                pieces.append((self._unk_token, self._unk_id))
                pos -= 1
                continue
            piece = text[start:pos]
            piece_id = self._pieces.piece_to_id(piece)
            if piece_id is None:
                piece_id = self._unk_id
                piece = self._unk_token
            pieces.append((piece, piece_id))
            pos = start

        pieces.reverse()
        return pieces

    def get_vocab(self) -> dict[str, int]:
        """Return piece → ID as a dict."""
        return {
            piece: idx
            for idx, (piece, _) in enumerate(self._pieces.pieces)
        }

    def save(self, directory: str | Path) -> None:
        """Save piece table to *directory*.

        Args:
            directory: Target directory.
        """
        save_pieces(self._pieces, directory)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        unk_token: str = UNK_TOKEN,
        unk_id: int = 0,
    ) -> "UnigramModel":
        """Load a :class:`UnigramModel` from *directory*.

        Args:
            directory: Source directory.
            unk_token: Unknown token string.
            unk_id: Unknown token ID.

        Returns:
            Loaded :class:`UnigramModel`.
        """
        pieces = load_pieces(directory)
        return cls(pieces, unk_token=unk_token, unk_id=unk_id)
