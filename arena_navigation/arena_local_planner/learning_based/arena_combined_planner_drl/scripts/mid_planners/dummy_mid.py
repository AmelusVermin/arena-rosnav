import numpy as np
from nav_msgs.msg import Odometry
import nav_msgs
from geometry_msgs.msg import PoseStamped
from .mid_planner import MidPlanner

NAME = "dummy mid planner"
class Dummy(MidPlanner):

    def __init__(self, ns):
        super().__init__(ns)
    
    @staticmethod
    def get_name() -> str:
        return NAME

    def get_subgoal(self, global_plan, odom):
        subgoal = PoseStamped()
        subgoal = global_plan.poses[-1]
        return subgoal

    def close(self):
        pass
    
    def is_ready(self):
        return True

    def reset(self):
        pass