import logging
import sys


def get_logger(name="MilvusVectorDB", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
