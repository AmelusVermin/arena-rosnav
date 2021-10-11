from typing import Callable, Type, Union

from stable_baselines3.common.policies import BasePolicy

from .base_agent import BaseAgent, check_format


class AgentFactory:
    """The factory class for creating agents"""

    registry = {}
    # registry for agent types without a specific callback
    no_cb_registry = []
    """ Internal registry for available agents """

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register agent class to the internal registry.

        Args:
            name (str): The name of the agent.

        Returns:
            The agent class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            assert name not in cls.registry, f"Agent '{name}' already exists!"
            assert issubclass(wrapped_class, BaseAgent) or issubclass(
                wrapped_class, BasePolicy
            ), f"Wrapped class {wrapped_class.__name__} is neither of type 'BaseAgent' nor 'BasePolicy!'"

            if issubclass(wrapped_class, BaseAgent):
                check_format(wrapped_class)

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    # end register()

    @classmethod
    def instantiate(cls, name: str, **kwargs) -> Union[Type[BaseAgent], Type[BasePolicy]]:
        """Factory command to create the agent.
        This method gets the appropriate agent class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        Args:
            name (str): The name of the agent to create.

        Returns:
            An instance of the agent that is created.
        """
        assert name in cls.registry, f"Agent '{name}' is not registered with callback!"
        agent_class = cls.registry[name]
        
        if issubclass(agent_class, BaseAgent):
            return agent_class(**kwargs)
        else:
            return agent_class

    @classmethod
    def register_no_cb(cls, name: str):
        """ 
            for registering agent types that are made with parameters in sb3 only
            the purpose is only for printing available types for configuration
        """
        if name not in cls.no_cb_registry:
            cls.no_cb_registry.append(name)

    @classmethod
    def get_all_registered_agents(cls):
        all_types = list(cls.registry.keys())
        all_types += cls.no_cb_registry        
        return all_types