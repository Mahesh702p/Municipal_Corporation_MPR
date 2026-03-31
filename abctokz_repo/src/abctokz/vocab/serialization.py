# Augenblick — abctokz
"""Artifact serialization helpers for vocab components."""

from __future__ import annotations

from pathlib import Path

from abctokz.constants import MERGES_FILENAME, PIECES_FILENAME, VOCAB_FILENAME
from abctokz.utils.io import load_json, save_json
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.pieces import PieceTable
from abctokz.vocab.vocab import Vocabulary


def save_vocab(vocab: Vocabulary, directory: str | Path) -> None:
    """Save a :class:`~abctokz.vocab.vocab.Vocabulary` to *directory*.

    Args:
        vocab: Vocabulary to save.
        directory: Target directory path.
    """
    save_json(vocab.to_dict(), Path(directory) / VOCAB_FILENAME)


def load_vocab(directory: str | Path, unk_token: str | None = None) -> Vocabulary:
    """Load a :class:`~abctokz.vocab.vocab.Vocabulary` from *directory*.

    Args:
        directory: Directory containing ``vocab.json``.
        unk_token: Unknown token string to use.

    Returns:
        Loaded :class:`~abctokz.vocab.vocab.Vocabulary`.
    """
    data = load_json(Path(directory) / VOCAB_FILENAME)
    return Vocabulary.from_dict(data, unk_token=unk_token)


def save_merges(table: MergeTable, directory: str | Path) -> None:
    """Save a :class:`~abctokz.vocab.merges.MergeTable` to *directory*.

    Saves in both text (``merges.txt``) for human readability.

    Args:
        table: Merge table to save.
        directory: Target directory path.
    """
    path = Path(directory) / MERGES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table.to_text(), encoding="utf-8")


def load_merges(directory: str | Path) -> MergeTable:
    """Load a :class:`~abctokz.vocab.merges.MergeTable` from *directory*.

    Args:
        directory: Directory containing ``merges.txt``.

    Returns:
        Loaded :class:`~abctokz.vocab.merges.MergeTable`.
    """
    path = Path(directory) / MERGES_FILENAME
    return MergeTable.from_text(path.read_text(encoding="utf-8"))


def save_pieces(table: PieceTable, directory: str | Path) -> None:
    """Save a :class:`~abctokz.vocab.pieces.PieceTable` to *directory*.

    Args:
        table: Piece table to save.
        directory: Target directory path.
    """
    save_json(table.to_list(), Path(directory) / PIECES_FILENAME)


def load_pieces(directory: str | Path) -> PieceTable:
    """Load a :class:`~abctokz.vocab.pieces.PieceTable` from *directory*.

    Args:
        directory: Directory containing ``pieces.json``.

    Returns:
        Loaded :class:`~abctokz.vocab.pieces.PieceTable`.
    """
    data = load_json(Path(directory) / PIECES_FILENAME)
    return PieceTable.from_list(data)
