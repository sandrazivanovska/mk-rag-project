"""Centralised logging via loguru."""

import sys
from loguru import logger


def get_logger(name: str = "mk-rag"):
    """Return a loguru logger bound with the given name."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/mk_rag_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )
    return logger.bind(name=name)
