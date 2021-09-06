import numpy as np
from global_planners.global_planner import GlobalPlanner
import nav_msgs

class Dummy(GlobalPlanner):

    def __init__(self):
        super().__init__()
        self.name = "dummy global planner"

    def get_name(self):
        return self.name

    def get_global_plan(self, goal, odom):
        global_path = nav_msgs.msg.Path()
        global_path.poses.append(goal)
        return global_path
