"""
This is a centralized path management script for the av_perception project.
"""

import os
from pathlib import Path

from av_perception.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """ Find the project root by walking up until pyproject.toml is found """

    current = Path(__file__).resolve().parent

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find project root pyproject.toml in any parent directory")


PROJECT_ROOT = get_project_root()


def get_data_dir() -> Path:
    """ Get data directory, checking for environment variable override. """
    
    env_root = os.environ.get("AV_DATA_ROOT")
    
    if env_root:
        data_dir = Path(env_root)
        logger.debug(f"Data dir from AV_DATA_ROOT: {data_dir}")
    else:
        data_dir = PROJECT_ROOT/ "data"
        logger.debug(f"Data dir (default): {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_output_dir(experiment_name:str = "") -> Path:
    """ get or create an experiment output directory. Returns a path to the output directory """

    from datetime import datetime

    if not experiment_name:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = PROJECT_ROOT / "outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "checkpoints").mkdir(exist_ok = True)

    (output_dir / "logs").mkdir(exist_ok = True)

    (output_dir / "visualizations").mkdir(exist_ok = True)

    logger.info(f"output directory: {output_dir}")
    return output_dir


