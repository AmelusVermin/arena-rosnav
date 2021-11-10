import numpy as np
import nav_msgs
from abc import ABC, abstractmethod

class GlobalPlanner(ABC):

    def __init__(self, ns):
        self.ns = ns
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @abstractmethod
    def get_global_plan(self, goal, odom) -> nav_msgs.msg.Path:
        pass