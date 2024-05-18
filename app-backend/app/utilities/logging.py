import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

console = Console(color_system="256", width=150, style="green")


@lru_cache
def get_logger(module_name: str) -> logging.Logger:
    """
    Args:
        module_name (str): file name

    Returns:
        logging.logger: logger object
    """
    logger = logging.getLogger(module_name)
    handler = RichHandler(
        rich_tracebacks=True, console=console, tracebacks_show_locals=True
    )
    handler.setFormatter(
        logging.Formatter(
            "%(name)s - [ %(threadName)s:%(funcName)s:%(lineno)d ] - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
