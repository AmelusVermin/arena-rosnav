import numpy as np
from mid_planners.mid_planner import MidPlanner

class Dummy(MidPlanner):

    def __init__(self):
        super().__init__()
        self.name = "dummy mid planner"
        
    def get_name(self) -> str:
        return self.name

    def get_subgoal(self, subgoal):
        return np.array(planned_path[-1])
