import numpy as np
from abc import ABC, abstractmethod

class GlobalPlanner(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def plan_path(self, goal, odom) -> np.ndarray:
        pass

    @abstractmethod
    def init(self):
        pass
