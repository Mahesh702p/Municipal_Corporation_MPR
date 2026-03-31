# Augenblick — abctokz
"""BPE (Byte-Pair Encoding) tokenization model."""

from __future__ import annotations

from pathlib import Path

from abctokz.constants import BPE_CONTINUATION_PREFIX, UNK_TOKEN
from abctokz.models.base import Model
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.serialization import load_merges, load_vocab, save_merges, save_vocab
from abctokz.vocab.vocab import Vocabulary


class BPEModel(Model):
    """BPE tokenization model.

    Given a vocabulary and a ranked list of merge rules, segments each
    pre-token using the BPE algorithm: iteratively merges the highest-ranked
    pair until no more merges apply.

    Non-initial subword pieces are prefixed with ``continuation_prefix``
    (default ``"##"``) to distinguish continuation pieces from word-initial
    pieces.

    Args:
        vocab: Vocabulary mapping token strings to IDs.
        merges: :class:`~abctokz.vocab.merges.MergeTable` of learned merge rules.
        unk_token: Unknown token string.
        continuation_prefix: Prefix added to non-initial BPE pieces.
        end_of_word_suffix: Suffix added to the last piece of each word.

    Example::

        from abctokz.vocab import Vocabulary, MergeTable
        vocab = Vocabulary({"<unk>": 0, "h": 1, "e": 2, "he": 3, "##l": 4, "##lo": 5})
        merges = MergeTable([(("h", "e"), "he"), (("##l", "##o"), "##lo")])
        model = BPEModel(vocab, merges)
        result = model.tokenize("hello")
        # -> [("he", 3), ("##l", 4), ("##o", ...)] or similar
    """

    def __init__(
        self,
        vocab: Vocabulary,
        merges: MergeTable,
        unk_token: str = UNK_TOKEN,
        continuation_prefix: str = BPE_CONTINUATION_PREFIX,
        end_of_word_suffix: str = "",
    ) -> None:
        self._vocab = vocab
        self._merges = merges
        self._unk_token = unk_token
        self._cont_prefix = continuation_prefix
        self._eow_suffix = end_of_word_suffix

    def tokenize(self, sequence: str) -> list[tuple[str, int]]:
        """Tokenize *sequence* using BPE merge rules.

        Args:
            sequence: A single pre-token string.

        Returns:
            List of ``(token_string, token_id)`` pairs.
        """
        if not sequence:
            return []

        # Initialise: each character is a piece, mark non-initial with prefix
        pieces = self._init_pieces(sequence)

        # Apply merges greedily by rank
        pieces = self._apply_merges(pieces)

        # Map to IDs
        result: list[tuple[str, int]] = []
        unk_id = self._vocab.unk_id or 0
        for piece in pieces:
            token_id = self._vocab._vocab.get(piece, unk_id)
            result.append((piece, token_id))
        return result

    def _init_pieces(self, word: str) -> list[str]:
        """Break *word* into initial character pieces with continuation prefix.

        Args:
            word: Raw word string.

        Returns:
            List of initial pieces.
        """
        chars = list(word)
        if not chars:
            return []
        pieces = [chars[0]]
        for ch in chars[1:]:
            pieces.append(self._cont_prefix + ch)
        if self._eow_suffix and pieces:
            pieces[-1] = pieces[-1] + self._eow_suffix
        return pieces

    def _apply_merges(self, pieces: list[str]) -> list[str]:
        """Iteratively apply the highest-priority merge rule.

        Args:
            pieces: Current list of pieces.

        Returns:
            Pieces after all applicable merges have been applied.
        """
        while len(pieces) > 1:
            best_rank: int | None = None
            best_idx = -1

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self._merges.get_rank(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # no applicable merge

            # Perform the merge
            merged = self._merges.merge_result((pieces[best_idx], pieces[best_idx + 1]))
            if merged is None:
                break
            pieces = pieces[:best_idx] + [merged] + pieces[best_idx + 2:]

        return pieces

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary dict."""
        return self._vocab.to_dict()

    def save(self, directory: str | Path) -> None:
        """Save vocabulary and merge rules to *directory*.

        Args:
            directory: Target directory.
        """
        save_vocab(self._vocab, directory)
        save_merges(self._merges, directory)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        unk_token: str = UNK_TOKEN,
        continuation_prefix: str = BPE_CONTINUATION_PREFIX,
        end_of_word_suffix: str = "",
    ) -> "BPEModel":
        """Load a :class:`BPEModel` from *directory*.

        Args:
            directory: Source directory.
            unk_token: Unknown token string.
            continuation_prefix: Continuation subword prefix.
            end_of_word_suffix: End-of-word suffix.

        Returns:
            Loaded :class:`BPEModel`.
        """
        vocab = load_vocab(directory, unk_token=unk_token)
        merges = load_merges(directory)
        return cls(
            vocab,
            merges,
            unk_token=unk_token,
            continuation_prefix=continuation_prefix,
            end_of_word_suffix=end_of_word_suffix,
        )
