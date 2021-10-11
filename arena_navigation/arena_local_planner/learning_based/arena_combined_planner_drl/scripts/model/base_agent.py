from typing import Type, List

from abc import ABC, abstractmethod
from enum import Enum
from torch.nn.modules.module import Module
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PolicyType(Enum):
    CNN = "CnnPolicy"
    MLP = "MlpPolicy"


class BaseAgent(ABC):
    """Base class for models loaded on runtime from
    the Stable-Baselines3 policy registry during PPO instantiation.
    The architecture of the eventual policy is determined by the
    'policy_kwargs' of the SB3 RL algorithm.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def type(self) -> PolicyType:
        pass

    @property
    @abstractmethod
    def features_extractor_class(self) -> Type[BaseFeaturesExtractor]:
        pass

    @property
    @abstractmethod
    def features_extractor_kwargs(self) -> dict:
        pass

    @property
    @abstractmethod
    def net_arch(self) -> List[dict]:
        pass

    @property
    @abstractmethod
    def activation_fn(self) -> Type[Module]:
        pass

    def get_kwargs(self):
        kwargs = {
            "features_extractor_class": self.features_extractor_class,
            "features_extractor_kwargs": self.features_extractor_kwargs,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
        }
        if not kwargs['features_extractor_class']:
            del kwargs['features_extractor_class']
        if not kwargs['features_extractor_kwargs']:
            del kwargs['features_extractor_kwargs']
        return kwargs

def check_format(cls: Type[BaseAgent]):
    
    assert isinstance(cls.type, PolicyType), "Type has to be of type 'PolicyType'!"
    if cls.features_extractor_class:
        assert issubclass(
            cls.features_extractor_class, BaseFeaturesExtractor
        ), "Feature extractors have to derive from 'BaseFeaturesExtractor'!"
    
    if cls.features_extractor_kwargs:
        assert (
            type(cls.features_extractor_kwargs) is dict
        ), "Features extractor kwargs have to be of type 'dict'!"

    if cls.net_arch:
        assert (
            type(cls.net_arch) is list
        ), "Network architecture kwargs have to be of type 'list'!"
        for entry in cls.net_arch:
            assert (
                type(entry) is dict or type(entry) is int
            ), "Network architecture entries have to be of either type 'list' or 'dict'!"
            if type(entry) is dict:
                assert "pi" in entry or "vf" in entry, (
                    "net_arch dictionaries have to contain either 'pi' or 'vf'"
                    "for the respective network head!"
                )

    if cls.activation_fn:
        assert issubclass(
            cls.activation_fn, Module
        ), "Activation functions have to be taken from torch!"

