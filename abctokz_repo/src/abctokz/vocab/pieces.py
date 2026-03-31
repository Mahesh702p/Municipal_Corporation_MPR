# Augenblick — abctokz
"""Unigram piece table: stores pieces and their log-probabilities."""

from __future__ import annotations

import math

from abctokz.types import PieceScore


class PieceTable:
    """Stores Unigram pieces with their log-probability scores.

    Pieces are indexed by their string for O(1) score lookup.

    Args:
        pieces: Ordered list of ``(piece, log_prob)`` tuples.  Pieces are
            expected to be sorted by ID (index = ID).

    Example::

        table = PieceTable([("▁hello", -3.0), ("▁world", -3.5), ("<unk>", 0.0)])
        assert table.score("▁hello") == -3.0
        assert table.piece_to_id("▁world") == 1
    """

    def __init__(self, pieces: list[PieceScore]) -> None:
        self._pieces: list[PieceScore] = list(pieces)
        self._piece_to_id: dict[str, int] = {piece: idx for idx, (piece, _) in enumerate(pieces)}
        self._scores: dict[str, float] = {piece: score for piece, score in pieces}

    @property
    def pieces(self) -> list[PieceScore]:
        """All ``(piece, score)`` pairs."""
        return list(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def score(self, piece: str) -> float | None:
        """Return the log-probability score of *piece*, or ``None``.

        Args:
            piece: Piece string.

        Returns:
            Log-probability score, or ``None`` if not in table.
        """
        return self._scores.get(piece)

    def piece_to_id(self, piece: str) -> int | None:
        """Return the ID of *piece*, or ``None`` if not in table.

        Args:
            piece: Piece string.

        Returns:
            Integer ID, or ``None``.
        """
        return self._piece_to_id.get(piece)

    def id_to_piece(self, piece_id: int) -> str | None:
        """Return the piece string for *piece_id*, or ``None``.

        Args:
            piece_id: Integer ID.

        Returns:
            Piece string, or ``None``.
        """
        if 0 <= piece_id < len(self._pieces):
            return self._pieces[piece_id][0]
        return None

    def __contains__(self, piece: str) -> bool:
        return piece in self._piece_to_id

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> list[list[object]]:
        """Serialise to a list of ``[piece, score]`` pairs."""
        return [[piece, score] for piece, score in self._pieces]

    @classmethod
    def from_list(cls, data: list[list[object]]) -> "PieceTable":
        """Deserialise from a list of ``[piece, score]`` pairs.

        Args:
            data: Serialised piece list.

        Returns:
            New :class:`PieceTable`.
        """
        pieces: list[PieceScore] = [(str(row[0]), float(row[1])) for row in data]
        return cls(pieces)
