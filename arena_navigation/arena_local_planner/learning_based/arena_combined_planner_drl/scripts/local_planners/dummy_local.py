import numpy as np
import torch as th
from local_planners.local_planner import LocalPlanner

class Dummy(LocalPlanner):

    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self.name = "dummy local planner"

    def get_name(self) -> str:
        return self.name

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations