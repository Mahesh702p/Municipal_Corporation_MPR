# Augenblick — abctokz
"""Unigram language model trainer.

Implements an EM-style Unigram training algorithm:
1. Build a large seed vocabulary from all character n-grams up to
   ``max_piece_length``.
2. Run EM iterations:
   a. E-step: Viterbi-segment the corpus, accumulate piece expected counts.
   b. M-step: Update piece log-probs from counts; compute total loss.
   c. Prune pieces with the lowest marginal contribution to the loss until
      the vocab shrinks to ``vocab_size * shrinking_factor``.
3. Repeat until vocabulary size <= ``vocab_size``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterator

from abctokz.config.schemas import UnigramTrainerConfig
from abctokz.models.unigram import UnigramModel
from abctokz.trainers.base import Trainer
from abctokz.types import PieceScore
from abctokz.utils.logging import get_logger
from abctokz.utils.seeds import set_seed
from abctokz.vocab.pieces import PieceTable

logger = get_logger(__name__)

NEG_INF = -1e9


def _viterbi_segment(
    text: str,
    scores: dict[str, float],
    max_len: int,
    unk_score: float,
) -> list[str]:
    """Segment *text* using Viterbi given piece *scores*.

    Args:
        text: Input string.
        scores: Piece → log-prob mapping.
        max_len: Maximum piece length to consider.
        unk_score: Log-prob assigned to any unknown single char.

    Returns:
        List of piece strings forming the best segmentation.
    """
    n = len(text)
    best = [NEG_INF] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0

    for end in range(1, n + 1):
        for start in range(max(0, end - max_len), end):
            if best[start] == NEG_INF:
                continue
            piece = text[start:end]
            sc = scores.get(piece, None)
            if sc is None:
                if end - start == 1:
                    sc = unk_score
                else:
                    continue
            cand = best[start] + sc
            if cand > best[end]:
                best[end] = cand
                back[end] = start

    # Backtrack
    pieces: list[str] = []
    pos = n
    while pos > 0:
        start = back[pos]
        if start < 0:
            pieces.append(text[pos - 1])
            pos -= 1
        else:
            pieces.append(text[start:pos])
            pos = start
    pieces.reverse()
    return pieces


class UnigramTrainer(Trainer):
    """Trains a :class:`~abctokz.models.unigram.UnigramModel`.

    Args:
        config: Unigram trainer configuration.
    """

    def __init__(self, config: UnigramTrainerConfig) -> None:
        self._config = config
        set_seed(config.seed)

    def train(self, corpus: Iterator[str]) -> UnigramModel:
        """Train Unigram model from *corpus*.

        Args:
            corpus: Iterable of text sentences.

        Returns:
            Trained :class:`~abctokz.models.unigram.UnigramModel`.
        """
        cfg = self._config
        logger.info("Unigram training: target vocab_size=%d", cfg.vocab_size)

        # Collect all training words with their frequencies
        word_freq: Counter[str] = Counter()
        for line in corpus:
            for word in line.split():
                word_freq[word] += 1

        # Build seed vocabulary: all substrings up to max_piece_length
        seed_vocab = self._build_seed_vocab(word_freq, cfg.max_piece_length)
        logger.info("Seed vocab size: %d", len(seed_vocab))

        # Initial piece scores (uniform log-prob)
        scores: dict[str, float] = {p: -math.log(len(seed_vocab)) for p in seed_vocab}
        for sp in cfg.special_tokens:
            scores[sp] = 0.0

        max_len = cfg.max_piece_length

        # EM loop with pruning
        for iteration in range(cfg.n_sub_iterations * 4):
            # E-step: compute expected counts via Viterbi
            counts: Counter[str] = Counter()
            total_tokens = 0
            unk_score = math.log(1.0 / max(len(scores), 1))
            for word, freq in word_freq.items():
                pieces = _viterbi_segment(word, scores, max_len, unk_score)
                for p in pieces:
                    counts[p] += freq
                    total_tokens += freq

            if total_tokens == 0:
                break

            # M-step: update scores
            log_total = math.log(total_tokens)
            new_scores: dict[str, float] = {}
            for piece in scores:
                cnt = counts.get(piece, 0)
                if cnt > 0:
                    new_scores[piece] = math.log(cnt) - log_total
                else:
                    new_scores[piece] = NEG_INF
            # Always keep special tokens
            for sp in cfg.special_tokens:
                new_scores[sp] = 0.0
            scores = new_scores

            # Prune: remove pieces that contribute little
            current_size = len(scores)
            if current_size > cfg.vocab_size:
                target_size = max(
                    cfg.vocab_size,
                    int(current_size * cfg.shrinking_factor),
                )
                scores = self._prune(scores, target_size, cfg.special_tokens, word_freq, max_len)
                logger.debug(
                    "Unigram iter %d: vocab %d -> %d", iteration, current_size, len(scores)
                )

            if len(scores) <= cfg.vocab_size:
                break

        # Final pruning to exact target size
        if len(scores) > cfg.vocab_size:
            scores = self._prune(scores, cfg.vocab_size, cfg.special_tokens, word_freq, max_len)

        # Build piece table: special tokens first, rest sorted by score desc
        pieces = self._build_piece_list(scores, cfg.special_tokens)
        logger.info("Unigram trained: vocab=%d", len(pieces))

        unk_id = 0
        return UnigramModel(
            PieceTable(pieces),
            unk_token=cfg.unk_token,
            unk_id=unk_id,
        )

    def _build_seed_vocab(self, word_freq: Counter[str], max_piece_len: int) -> set[str]:
        """Extract all substrings up to *max_piece_len* with sufficient coverage."""
        cfg = self._config
        char_total = sum(len(w) * f for w, f in word_freq.items())
        char_freq: Counter[str] = Counter()
        for word, freq in word_freq.items():
            for ch in word:
                char_freq[ch] += freq

        # Character coverage filter
        sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])
        covered = 0
        allowed_chars: set[str] = set()
        for ch, cnt in sorted_chars:
            covered += cnt
            allowed_chars.add(ch)
            if covered / max(char_total, 1) >= cfg.char_coverage:
                break

        # Build n-gram vocab
        seed: set[str] = set(allowed_chars)
        for word in word_freq:
            for start in range(len(word)):
                for end in range(start + 2, min(start + max_piece_len + 1, len(word) + 1)):
                    substr = word[start:end]
                    if all(c in allowed_chars for c in substr):
                        seed.add(substr)

        return seed

    def _prune(
        self,
        scores: dict[str, float],
        target_size: int,
        special_tokens: list[str],
        word_freq: Counter[str],
        max_len: int,
    ) -> dict[str, float]:
        """Prune vocabulary to *target_size*.

        Removes pieces with the lowest scores, always keeping special tokens
        and individual characters (to guarantee coverage).

        Args:
            scores: Current piece scores.
            target_size: Target number of pieces.
            special_tokens: Pieces that must never be removed.
            word_freq: Word frequencies for loss computation.
            max_len: Max piece length.

        Returns:
            Pruned scores dict.
        """
        # Must-keep: special tokens + single chars
        must_keep: set[str] = set(special_tokens)
        for piece in scores:
            if len(piece) == 1:
                must_keep.add(piece)

        # Sort prunable pieces by score (ascending = remove first)
        prunable = sorted(
            [(p, s) for p, s in scores.items() if p not in must_keep],
            key=lambda x: x[1],
        )
        keep_count = target_size - len(must_keep)
        keep_pieces = {p for p, _ in prunable[-max(keep_count, 0):]}
        keep_pieces.update(must_keep)

        return {p: s for p, s in scores.items() if p in keep_pieces}

    def _build_piece_list(
        self, scores: dict[str, float], special_tokens: list[str]
    ) -> list[PieceScore]:
        """Build the final ordered piece list.

        Special tokens come first (in config order), then remaining pieces
        sorted by descending score for determinism.

        Args:
            scores: Final piece scores.
            special_tokens: Pieces to place first.

        Returns:
            Ordered list of ``(piece, score)`` tuples.
        """
        pieces: list[PieceScore] = []
        seen: set[str] = set()
        for sp in special_tokens:
            sc = scores.get(sp, 0.0)
            pieces.append((sp, sc))
            seen.add(sp)
        rest = sorted(
            [(p, s) for p, s in scores.items() if p not in seen],
            key=lambda x: (-x[1], x[0]),  # desc score, asc lex for ties
        )
        pieces.extend(rest)
        return pieces
