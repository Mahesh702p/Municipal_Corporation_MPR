# Augenblick — abctokz
"""BPE trainer: learns merge rules from corpus statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterator

from abctokz.config.schemas import BPETrainerConfig
from abctokz.models.bpe import BPEModel
from abctokz.trainers.base import Trainer
from abctokz.types import MergePair, MergeRules
from abctokz.utils.logging import get_logger
from abctokz.utils.seeds import set_seed
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.vocab import Vocabulary

logger = get_logger(__name__)

# A "word" in BPE training is represented as a tuple of pieces
Word = tuple[str, ...]
WordFreq = dict[Word, int]
PairFreq = dict[MergePair, int]


def _get_pair_freqs(word_freqs: WordFreq) -> PairFreq:
    """Count all adjacent-pair frequencies across the corpus.

    Args:
        word_freqs: Mapping from (piece, ...) tuple to frequency.

    Returns:
        Mapping from (left_piece, right_piece) to total frequency.
    """
    pairs: PairFreq = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_pair(pair: MergePair, word_freqs: WordFreq) -> WordFreq:
    """Apply one merge rule to all words, returning updated word_freqs.

    Args:
        pair: The ``(left, right)`` pair to merge.
        word_freqs: Current word piece frequencies.

    Returns:
        Updated word frequencies with the merge applied.
    """
    a, b = pair
    merged = a + b
    new_freqs: WordFreq = {}
    for word, freq in word_freqs.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = freq
    return new_freqs


class BPETrainer(Trainer):
    """Trains a :class:`~abctokz.models.bpe.BPEModel` using BPE.

    Algorithm:
    1. Build the initial alphabet (all characters in corpus).
    2. Represent each word as a sequence of character pieces (with
       continuation prefix on non-initial chars).
    3. Repeatedly find the most frequent adjacent pair and merge it,
       recording the merge rule.
    4. Stop when the target vocabulary size is reached.

    Tie-breaking in pair selection is deterministic: when multiple pairs
    have equal frequency, the lexicographically smallest pair is chosen.

    Args:
        config: BPE trainer configuration.
    """

    def __init__(self, config: BPETrainerConfig) -> None:
        self._config = config
        set_seed(config.seed)

    def train(self, corpus: Iterator[str]) -> BPEModel:
        """Learn BPE merge rules from *corpus*.

        Args:
            corpus: Iterable of sentences.

        Returns:
            Trained :class:`~abctokz.models.bpe.BPEModel`.
        """
        cfg = self._config
        prefix = cfg.continuing_subword_prefix
        eow = cfg.end_of_word_suffix
        logger.info("BPE training: target vocab_size=%d, min_freq=%d", cfg.vocab_size, cfg.min_frequency)

        # --- Step 1: Count word frequencies ---
        raw_freq: Counter[str] = Counter()
        for line in corpus:
            for word in line.split():
                raw_freq[word] += 1

        # Filter by min_frequency
        raw_freq = Counter({w: c for w, c in raw_freq.items() if c >= cfg.min_frequency})

        # --- Step 2: Build initial word representations ---
        word_freqs: WordFreq = {}
        alphabet: set[str] = set()

        for word, freq in raw_freq.items():
            chars = list(word)
            if not chars:
                continue
            pieces: list[str] = [chars[0]]
            for ch in chars[1:]:
                pieces.append(prefix + ch)
            if eow and pieces:
                pieces[-1] += eow
            for piece in pieces:
                alphabet.add(piece)
            word_freqs[tuple(pieces)] = freq

        # Apply alphabet limit if configured
        if cfg.limit_alphabet is not None:
            # Keep only the most frequent alphabet characters
            char_freq: Counter[str] = Counter()
            for word, freq in raw_freq.items():
                for ch in word:
                    char_freq[ch] += freq
            allowed_chars = {ch for ch, _ in char_freq.most_common(cfg.limit_alphabet)}
            # Re-filter words and alphabet
            alphabet = {p for p in alphabet if p.lstrip(prefix) in allowed_chars or p in cfg.special_tokens}

        # Add initial alphabet chars to allowed set
        initial = set(cfg.initial_alphabet)
        alphabet.update(initial)

        # --- Step 3: Special tokens always in vocab ---
        n_special = len(cfg.special_tokens)
        # We need (vocab_size - n_special - alphabet_size) merges
        target_merges = cfg.vocab_size - n_special - len(alphabet)
        target_merges = max(0, target_merges)

        merge_rules: MergeRules = []
        vocab_set: set[str] = set(alphabet)

        logger.info("Initial alphabet size: %d, target merges: %d", len(alphabet), target_merges)

        # --- Step 4: Learn merge rules ---
        for step in range(target_merges):
            pair_freqs = _get_pair_freqs(word_freqs)
            if not pair_freqs:
                break

            # Pick best pair: highest freq, then lex smallest for tie-breaking
            best_pair = max(
                pair_freqs.keys(),
                key=lambda p: (pair_freqs[p], (-ord(p[0][0]) if p[0] else 0)),
            )
            # Deterministic tie-breaking: sort by pair strings
            max_freq = pair_freqs[best_pair]
            candidates = [p for p, f in pair_freqs.items() if f == max_freq]
            candidates.sort()  # lexicographic determinism
            best_pair = candidates[0]

            b = best_pair[1]
            merged = best_pair[0] + (b[len(prefix):] if prefix and b.startswith(prefix) else b)
            merge_rules.append((best_pair, merged))
            vocab_set.add(merged)
            word_freqs = _merge_pair(best_pair, word_freqs)

            if step % 500 == 0:
                logger.debug("BPE step %d/%d: merged %r + %r -> %r", step, target_merges, *best_pair, merged)

        # --- Step 5: Build final vocabulary ---
        vocab: dict[str, int] = {}
        for i, sp in enumerate(cfg.special_tokens):
            vocab[sp] = i
        offset = n_special
        # Sort vocab_set deterministically
        for token in sorted(vocab_set):
            if token not in vocab:
                vocab[token] = offset
                offset += 1

        logger.info("BPE trained: vocab=%d, merges=%d", len(vocab), len(merge_rules))
        unk_token = cfg.special_tokens[0] if cfg.special_tokens else None
        return BPEModel(
            Vocabulary(vocab, unk_token=unk_token),
            MergeTable(merge_rules),
            unk_token=unk_token or "<unk>",
            continuation_prefix=prefix,
            end_of_word_suffix=eow,
        )
