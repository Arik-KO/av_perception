"""
CARLA simulator connection utility. Handles connecting to the CARLA server with proper
error handling, logging, and configuration.
"""


import os
import time
from typing import Optional

from av_perception.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import carla
except ImportError:
    raise ImportError("CARLA python api not found. Install with: pip install carla")


def connect( host: Optional[str] = None,
    port : int = 2000,
    timeout: float = 10.0,
    retries: int = 3,) -> "carla.Client" :
    """ connect to a running CARLA server.
    Args:
        host: CARLA server IP. If None, checks CARLA_HOST env var, then falls back to the localhost.
        port: CARLA server port. (default 2000).
        timeout: Connection timeout in seconds.
        retries: Number of connection attempts before giving up.

    returns: A connected carla.Client instance
    raises: RuntimeError: If connection fails after all retries."""

    if host is None:
        host = os.environ.get("CARLA_HOST", "localhost")

    logger.info(f"Connecting to CARLA at {host}:{port}")

    for attempt in range(1, retries+1):
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)

            world = client.get_world()
            map_name = world.get_map().name

            logger.info(f"connected to CARLA (attempt {attempt}/{retries})")
            logger.info(f"Map: {map_name}")
            logger.info(f"available vehicles: "
                    f"{len(world.get_blueprint_library().filter('vehicle.*'))}")
            return client

        except RuntimeError as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                wait_time = attempt * 2
                logger.info(f"Retrying in {wait_time}s ...")
                time.sleep(wait_time)

    raise RuntimeError(
        f"could not connect to CARLA at {host}:{port} after {retries} attempt. \n"
        f"make sure CARLA server is running.")
