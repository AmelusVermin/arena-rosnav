from typing import Callable
from stable_baselines.common.policies import BasePolicy, register_policy

class PolicyRegistry:
    """ Class for registering custom policies in stable baselines"""
    internal_registry = set()

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register policy class to the stable baselines registry.
        
        Args:
            name (str): The name of the policy.

        Returns:
            The policy class itself.
        """
        
        def inner_wrapper(wrapped_class) -> Callable:
            assert issubclass(wrapped_class, BasePolicy), f"Wrapped class {wrapped_class.__name__} is not type of 'BasePolicy'!"
            register_policy(name, wrapped_class)
            cls.internal_registry.add(name)
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_all_registered_policies(cls):
        return list(cls.internal_registry)