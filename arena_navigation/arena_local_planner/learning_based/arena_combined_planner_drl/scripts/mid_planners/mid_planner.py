import numpy as np
import nav_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from abc import ABC, abstractmethod

class MidPlanner(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_subgoal(self, global_plan : nav_msgs.msg.Path, odom : Odometry) -> PoseStamped:
        pass
