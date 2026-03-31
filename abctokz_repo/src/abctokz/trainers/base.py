# Augenblick — abctokz
"""Abstract base class for all abctokz trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from abctokz.models.base import Model


class Trainer(ABC):
    """Abstract base for all tokenizer trainers.

    A trainer consumes a corpus (as an iterable of strings) and produces a
    trained :class:`~abctokz.models.base.Model`. Trainers are responsible for
    building the vocabulary and any model-specific artifacts (merge rules,
    piece scores, etc.).

    All trainers must be **deterministic**: the same corpus and the same seed
    must always produce the same model.

    Example::

        class MyTrainer(Trainer):
            def train(self, corpus: Iterator[str]) -> Model:
                ...
    """

    @abstractmethod
    def train(self, corpus: Iterator[str]) -> Model:
        """Train a model from *corpus*.

        Args:
            corpus: An iterable of strings (one sentence / line per item).

        Returns:
            A fully trained :class:`~abctokz.models.base.Model`.
        """

    def train_from_files(self, paths: list[str]) -> Model:
        """Convenience wrapper to train from a list of text file paths.

        Args:
            paths: List of paths to UTF-8 text files.

        Returns:
            Trained model.
        """
        def _iter() -> Iterator[str]:
            for path in paths:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield line

        return self.train(_iter())
