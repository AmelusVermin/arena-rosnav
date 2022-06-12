import nav_msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from abc import ABC, abstractmethod

class MidPlanner(ABC):

    def __init__(self, ns):
        self.ns = ns
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """ returns the name of the planner """
        pass

    @abstractmethod
    def get_subgoal(self, global_plan : nav_msgs.msg.Path, odom : Odometry) -> PoseStamped:
        """ returns the subgoal """
        pass

    @abstractmethod
    def close(self):
        """ clean up method that is called to close this planner. 
            It is called in the close() method of the environment 
        """
        pass

    @abstractmethod
    def is_ready(self):
        """ It is called after __init__ in a sim step loop until True is returned.
            The planner might need a couple of sim steps to finish initialization.
        """
        pass

    @abstractmethod
    def reset(self):
        """ The method is called in the reset() method of the environment. 
            The planner shall prepare for the next episode.
        """
        pass