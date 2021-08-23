import numpy as np
from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LocalPlanner(BaseFeaturesExtractor, ABC):

    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)

    @abstractmethod
    def get_name(self) -> str:
        pass

