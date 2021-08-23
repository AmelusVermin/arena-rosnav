import gym
import numpy as np
import rospy

from gym import spaces

class Observer():

    def __init__(self, args):
        self.observation_space = spaces.Tuple((
            # global goal
            spaces.Box(low=-args.max_distance_goal, high=args.max_distance_goal, shape=(7,), dtype=np.float32),
            # lidar scan
            spaces.Box(low=0, high=args.lidar_range, shape=(args.num_lidar_beams,), dtype=np.float32),
        ))
        #TODO init subscribers

    def get_observation_space(self):
        return self.observation_space

    def get_observation(self):
        """get observations from subscribed topics"""
        #TODO read subscribed topics
        return self.observation_space.sample() 

