import nav_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from abc import ABC, abstractmethod

class GlobalPlanner(ABC):

    def __init__(self, ns):
        self.ns = ns
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @abstractmethod
    def get_global_plan(self, goal: PoseStamped, odom: Odometry) -> nav_msgs.msg.Path:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_ready(self):
        pass