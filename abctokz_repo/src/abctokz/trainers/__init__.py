# Augenblick — abctokz
"""Trainer subpackage for abctokz."""

from abctokz.trainers.base import Trainer
from abctokz.trainers.bpe_trainer import BPETrainer
from abctokz.trainers.unigram_trainer import UnigramTrainer
from abctokz.trainers.wordlevel_trainer import WordLevelTrainer
from abctokz.config.schemas import BPETrainerConfig, TrainerConfig, UnigramTrainerConfig, WordLevelTrainerConfig


def build_trainer(config: TrainerConfig) -> Trainer:
    """Construct a :class:`Trainer` from a config object.

    Args:
        config: A validated trainer config.

    Returns:
        Corresponding :class:`Trainer` instance.

    Raises:
        ValueError: For unknown config types.
    """
    if isinstance(config, WordLevelTrainerConfig):
        return WordLevelTrainer(config)
    if isinstance(config, BPETrainerConfig):
        return BPETrainer(config)
    if isinstance(config, UnigramTrainerConfig):
        return UnigramTrainer(config)
    raise ValueError(f"Unknown trainer config type: {type(config)}")


__all__ = [
    "Trainer",
    "BPETrainer",
    "UnigramTrainer",
    "WordLevelTrainer",
    "build_trainer",
]

