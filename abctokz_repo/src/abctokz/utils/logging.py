# Augenblick — abctokz
"""Logging utilities for abctokz."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from abctokz.constants import LOG_DATE_FORMAT, LOG_FORMAT


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a module-level logger configured for abctokz.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Optional log level override; defaults to WARNING if not set
               by a parent logger.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root *abctokz* logger with a sensible stream handler.

    Safe to call multiple times; skips setup if handlers are already present.

    Args:
        level: Log level for the root abctokz logger.
    """
    root = logging.getLogger("abctokz")
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(handler)
