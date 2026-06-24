"""
centralized logging configuration for autonomous vehicle perception task.

"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logger( name: str = 'av_perception', log_dir : Optional[Path] = None, console_level:int = logging.INFO, file_level :int = logging.DEBUG,
) -> logging.Logger:

    """
    configure and return project root logger. 

    Arguments:
    name: Root logger name.
    log_dir : Directory for the log files. None means console only.
    console_level: minimum level shown in the terminal
    file_level :  minimum level written in the log file

    Returns:
    configured root logger

    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    #console handler
    console_formatter = logging.Formatter(
    fmt ="[%(asctime)s] %(levelname)-8s| %(name)-35s | %(message)s",
    datefmt ="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents = True, exist_ok = True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_formatter = logging.Formatter(
        fmt ="[%(asctime)s] %(levelname)-8s| %(name)-35s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt ="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file created: {log_file}")

    logger.propagate = False

    return logger



def get_logger(name: str) -> logging.Logger:
    """

    get a child logger for a specific module

    """
    return logging.getLogger(name)

