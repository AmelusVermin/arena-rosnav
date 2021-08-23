import numpy as np
from global_planners.global_planner import GlobalPlanner

class Dummy(GlobalPlanner):

    def __init__(self):
        super().__init__()
        self.name = "dummy global planner"

    def get_name(self):
        return self.name

    def plan_path(self, goal, odom):
        return np.array([odom, goal])
