"""Logging configs."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging with a specified level.

    Args:
        level (str): The logging level, e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR'.

    """
    level_dict = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}

    logging_level = level_dict.get(level.upper(), logging.INFO)

    logging.basicConfig(
        level=logging_level,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],  # Prints logs to the console
    )
