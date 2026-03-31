# Augenblick — abctokz
"""WordLevel trainer: builds a frequency-based vocabulary."""

from __future__ import annotations

from collections import Counter
from typing import Iterator

from abctokz.config.schemas import WordLevelTrainerConfig
from abctokz.models.wordlevel import WordLevelModel
from abctokz.trainers.base import Trainer
from abctokz.utils.logging import get_logger
from abctokz.utils.seeds import set_seed
from abctokz.vocab.vocab import Vocabulary

logger = get_logger(__name__)


class WordLevelTrainer(Trainer):
    """Trains a :class:`~abctokz.models.wordlevel.WordLevelModel`.

    Counts word frequencies over the corpus, retains the top-k words
    (where k = ``vocab_size`` - ``len(special_tokens)``), and assigns
    consecutive IDs. Special tokens are always assigned the lowest IDs.

    Ordering is **deterministic**: special tokens first (in config order),
    then words sorted by (descending frequency, ascending lexicographic order)
    to break ties reproducibly.

    Args:
        config: Trainer configuration.

    Example::

        from abctokz.config.schemas import WordLevelTrainerConfig
        trainer = WordLevelTrainer(WordLevelTrainerConfig(vocab_size=100))
        model = trainer.train(iter(["hello world", "hello"]))
    """

    def __init__(self, config: WordLevelTrainerConfig) -> None:
        self._config = config
        set_seed(config.seed)

    def train(self, corpus: Iterator[str]) -> WordLevelModel:
        """Build a vocabulary from *corpus*.

        Args:
            corpus: Iterable of sentences/lines.

        Returns:
            Trained :class:`~abctokz.models.wordlevel.WordLevelModel`.
        """
        cfg = self._config
        logger.info("WordLevel training: vocab_size=%d, min_freq=%d", cfg.vocab_size, cfg.min_frequency)

        freq: Counter[str] = Counter()
        for line in corpus:
            for token in line.split():
                freq[token] += 1

        # Determine capacity for non-special tokens
        n_special = len(cfg.special_tokens)
        capacity = cfg.vocab_size - n_special

        # Filter by min_frequency, then take top-k by (freq desc, lex asc)
        eligible = [(tok, cnt) for tok, cnt in freq.items() if cnt >= cfg.min_frequency]
        eligible.sort(key=lambda x: (-x[1], x[0]))
        selected = [tok for tok, _ in eligible[:capacity]]

        # Build vocab: special tokens first
        vocab: dict[str, int] = {}
        for i, sp in enumerate(cfg.special_tokens):
            vocab[sp] = i
        offset = n_special
        for tok in selected:
            if tok not in vocab:
                vocab[tok] = offset
                offset += 1

        logger.info("WordLevel trained vocab of size %d", len(vocab))
        return WordLevelModel(Vocabulary(vocab, unk_token=cfg.special_tokens[0] if cfg.special_tokens else None))
