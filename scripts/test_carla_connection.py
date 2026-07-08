""" quick test to verify if CARLA connection works. """

from av_perception.utils.logger import setup_logger
from av_perception.utils.carla_client import connect

logger = setup_logger()


client = connect(host= "write_host_address")
print("CARLA connection is working!")
