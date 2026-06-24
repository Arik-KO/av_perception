""" quick test to verify if CARLA connection works. """

from av_perception.utils.logger import setup_logger
from av_perception.utils.carla_client import connect

logger = setup_logger()


client = connect(host= "172.23.64.1")
print("CARLA connection is working!")
