import numpy as np
import nav_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from abc import ABC, abstractmethod

class MidPlanner(ABC):

    def __init__(self, ns):
        self.ns = ns
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @abstractmethod
    def get_subgoal(self, global_plan : nav_msgs.msg.Path, odom : Odometry) -> PoseStamped:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_ready(self):
        pass

    @abstractmethod
    def reset(self):
        pass