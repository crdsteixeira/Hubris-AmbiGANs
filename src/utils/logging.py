"""Logging configs"""

import logging

def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()  # Prints logs to the console
        ]
    )

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()  # Prints logs to the console
        ]
    )

def configure_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()  # Prints logs to the console
        ]
    )

def configure_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()  # Prints logs to the console
        ]
    )