import logging
import sys


def setup_logger(logger_name, log_level=logging.DEBUG):
    # Create a logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler and add it to the logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
