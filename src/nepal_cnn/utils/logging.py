"""Readable console/file logging for unattended pruning experiments."""

from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "nepal_cnn.pruning"


def configure_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not any(getattr(handler, "_nepal_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._nepal_console = True
        logger.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == resolved
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return configure_logging()
