import numpy as np
from abc import ABC, abstractmethod

class MidPlanner(ABC):

    def __init__(self):
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_subgoal(self, planned_path : np.ndarray) -> np.ndarray:
        pass
