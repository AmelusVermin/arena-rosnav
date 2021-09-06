import numpy as np
from nav_msgs.msg import Odometry
import nav_msgs
from geometry_msgs.msg import PoseStamped
from mid_planners.mid_planner import MidPlanner

class Dummy(MidPlanner):

    def __init__(self):
        super().__init__()
        self.name = "dummy mid planner"
        
    def get_name(self) -> str:
        return self.name

    def get_subgoal(self, global_plan, odom):
        subgoal = PoseStamped()
        subgoal = global_plan.poses[0]
        return subgoal
