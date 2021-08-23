import gym
import rospy
import numpy as np

from gym import spaces
from utils.observer import Observer
from global_planners.dummy_global import Dummy
class FlatlandEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, args, global_planner, mid_planner):
        super(FlatlandEnv, self).__init__()
        
        # observer related inits
        self.observer = Observer(args)
        self.observation_space = self.observer.get_observation_space()

        # action space of agent (local planner)
        self.action_space = spaces.Box(
            low=np.array([args.linear_range[0], args.angular_range[0]]),
            high=np.array([args.linear_range[1], args.angular_range[1]]),
            dtype=np.float,
        )

        # set global and mid planner
        self.global_planner = global_planner
        self.mid_planner = mid_planner
        
        # debug output
        rospy.logdebug(f"observation space: {self.observation_space}")
        rospy.logdebug(f"action space: {self.action_space}")
        

    def step(self, action):
        observation = self.observer.get_observation()
        reward = 0
        done = 0
        info = None
        return observation, reward, done, info
    
    def reset(self):
        observation = self.observation_space.sample()
        return observation  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        pass