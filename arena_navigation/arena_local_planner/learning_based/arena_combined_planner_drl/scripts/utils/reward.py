import numpy as np
import scipy.spatial
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.rl_agent.utils import reward
from geometry_msgs.msg import Pose2D
from typing import Tuple
from collections import deque
import rospy

class RewardCalculator:
    def __init__(
        self,
        robot_radius: float,
        safe_dist: float,
        goal_radius: float,
        collision_tolerance: float,
        rule: str = "rule_00",
        extended_eval: bool = False,
        max_timesteps: int = 500
    ):
        """
        A class for calculating reward based various rules.


        :param safe_dist (float): The minimum distance to obstacles or wall that robot is in safe status.
                                  if the robot get too close to them it will be punished. Unit[ m ]
        :param goal_radius (float): The minimum distance to goal that goal position is considered to be reached.
        """
        self.curr_reward = 0
        # additional info will be stored here and be returned alonge with reward.
        self.info = {}
        self._reward_composition = {}
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.last_goal_dist = None
        self.last_subgoal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self.last_global_plan = None
        self.safe_dist = safe_dist
        self.collision_tolerance = collision_tolerance
        self._extended_eval = extended_eval
        self._max_timesteps = max_timesteps
        self._prior_path_lengths = deque(maxlen=20)

        self.kdtree = None

        self._cal_funcs = {
            "rule_00": RewardCalculator._cal_reward_rule_00,
            "rule_01": RewardCalculator._cal_reward_rule_01,
            "rule_02": RewardCalculator._cal_reward_rule_02,
            "rule_03": RewardCalculator._cal_reward_rule_03,
            "rule_04": RewardCalculator._cal_reward_rule_04,
            "rule_05": RewardCalculator._cal_reward_rule_05,
        }
        self.cal_func = self._cal_funcs[rule]

    def reset(self):
        """
        reset variables related to the episode
        """
        self.last_goal_dist = None
        self.last_dist_to_path = None
        self.last_action = None
        self.kdtree = None
        self.last_global_plan = None
        self._prior_path_lengths.clear()

    def _reset(self):
        """
        reset variables related to current step
        """
        self.curr_reward = 0
        self.info = {}
        self._reward_composition = {
            "time consumption" : 0,
            "reduced path length" : 0,
            "goal reached" : 0,
            "goal approached" : 0,
            "collision" : 0,
            "safe dist" : 0,
            "not moving" : 0,
            "distance traveled" : 0,
            "distance global plan" : 0,
            "following global plan" : 0,
            "abrupt direction change" : 0
        }

    def get_reward(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        """
        Returns reward and info to the gym environment.

        :param laser_scan (np.ndarray): laser scan data
        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        """
        self._reset()
        self.cal_func(self, laser_scan, goal_in_robot_frame, *args, **kwargs)
        assert not np.isnan(self.curr_reward) and not np.isinf(self.curr_reward), "reward is nan or inf: {reward}"
        self.last_global_plan = kwargs["global_plan"]
        return self.curr_reward, self.info, self._reward_composition

    def _cal_reward_rule_00(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_goal_reached(goal_in_robot_frame)
        self._reward_safe_dist(laser_scan, punishment_factor=0.25)
        self._reward_collision(laser_scan)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )
        self.curr_reward = sum(self._reward_composition.values())

    def _cal_reward_rule_01(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_goal_reached(goal_in_robot_frame, reward_factor=15)
        self._reward_safe_dist(laser_scan, punishment_factor=0.25)
        self._reward_collision(laser_scan, punishment_factor=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )
        self.curr_reward = sum(self._reward_composition.values())

    def _cal_reward_rule_02(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_distance_traveled(kwargs["action"], consumption_factor=0.0075)
        self._reward_following_global_plan(kwargs["global_plan"], kwargs["robot_pose"])
        self._reward_goal_reached(goal_in_robot_frame, reward_factor=15)
        self._reward_safe_dist(laser_scan, punishment_factor=0.25)
        self._reward_collision(laser_scan, punishment_factor=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )
        self.curr_reward = sum(self._reward_composition.values())

    def _cal_reward_rule_03(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_following_global_plan(
            kwargs["global_plan"], kwargs["robot_pose"], kwargs["action"]
        )
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                kwargs["global_plan"],
                kwargs["robot_pose"],
                reward_factor=0.2,
                penalty_factor=0.3,
            )
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward_factor=15)
        self._reward_safe_dist(laser_scan, punishment_factor=0.25)
        self._reward_collision(laser_scan, punishment_factor=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )
        self.curr_reward = sum(self._reward_composition.values())

    def _cal_reward_rule_04(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        self._reward_abrupt_direction_change(kwargs["action"])
        self._reward_following_global_plan(
            kwargs["global_plan"], kwargs["robot_pose"], kwargs["action"]
        )
        if laser_scan.min() > self.safe_dist:
            self._reward_distance_global_plan(
                kwargs["global_plan"],
                kwargs["robot_pose"],
                reward_factor=0.2,
                penalty_factor=0.3,
            )
        else:
            self.last_dist_to_path = None
        self._reward_goal_reached(goal_in_robot_frame, reward_factor=15)
        self._reward_safe_dist(laser_scan, punishment_factor=0.25)
        self._reward_collision(laser_scan, punishment_factor=10)
        self._reward_goal_approached(
            goal_in_robot_frame, reward_factor=0.3, penalty_factor=0.4
        )
        self.curr_reward = sum(self._reward_composition.values())

    def _cal_reward_rule_05(
        self,
        laser_scan: np.ndarray,
        goal_in_robot_frame: Tuple[float, float],
        *args,
        **kwargs
    ):
        #self._reward_distance_traveled(kwargs["action"], consumption_factor=0.025)
        self._reward_abrupt_direction_change(kwargs["action"], punishment_weight=1000)
        self._reward_goal_reached(goal_in_robot_frame, reward_factor=45)
        self._reward_goal_approached(goal_in_robot_frame, reward_factor=0.8, penalty_factor=0.6)
        self._reward_safe_dist(laser_scan, punishment_factor=1.25)
        self._reward_collision(laser_scan, punishment_factor=50)
        #self._reward_reduced_path_length(kwargs["global_plan_length"], reward_factor=0.015)
        self._reward_following_global_plan(kwargs["global_plan"], kwargs["robot_pose"], action=kwargs["action"], dist_to_path=0.5, reward_factor=0.3)
        self._reward_distance_global_plan(kwargs["global_plan"], kwargs["robot_pose"], reward_factor=0.1, penalty_factor=0.1)
        self._reward_time_consumption(kwargs["episode_steps_passed"], max_punishment=30)
        rospy.logdebug(self._reward_composition)
        self.curr_reward = sum(self._reward_composition.values())

    
    def _reward_time_consumption(self, steps_passed, max_punishment=30):
        """ punishes the agent if it takes more and more timesteps to reach the goal based on """
        # calculate a factor based on "sum(i^2)_i=1_to_n = (n(n+1)(2n+1))/6"
        # the reward sum over all timesteps to max_timesteps shall be equal to max_punishment 
        factor = max_punishment / ((self._max_timesteps * (self._max_timesteps + 1) * (2 * self._max_timesteps + 1))/6)
        reward = -factor * steps_passed**2
        self._reward_composition["time consumption"] = reward
        return reward
    
    def _reward_reduced_path_length(self, current_length, reward_factor=1):
        reward = 0
        if self._prior_path_lengths:
            step = 1/len(self._prior_path_lengths)
            for i, prior_length in enumerate(self._prior_path_lengths):
                reward += (prior_length - current_length) * (i+1) * step 

        self._prior_path_lengths.append(current_length)
        reward *= reward_factor
        self._reward_composition["reduced path length"] = reward
        return reward


    def _reward_goal_reached(
        self, goal_in_robot_frame=Tuple[float, float], reward_factor: float = 15
    ):
        """
        Reward for reaching a given point.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward (float, optional): reward amount for reaching. defaults to 15
        """
        if goal_in_robot_frame[0] < self.goal_radius + self.robot_radius:
            reward = reward_factor
            self.info["is_done"] = True
            self.info["done_reason"] = 2
            self.info["is_success"] = 1
        else:
            self.info["is_done"] = False
            reward = 0
        self._reward_composition["goal reached"] = reward
        return reward
        
    def _reward_goal_approached(
        self,
        goal_in_robot_frame=Tuple[float, float],
        reward_factor: float = 0.3,
        penalty_factor: float = 0.5,
    ):
        """
        Reward for approaching the goal.

        :param goal_in_robot_frame (Tuple[float,float]): position (rho, theta) of the goal in robot frame (Polar coordinate)
        :param reward_factor (float, optional): positive factor for approaching goal. defaults to 0.3
        :param penalty_factor (float, optional): negative factor for withdrawing from goal. defaults to 0.5
        """
        reward = 0
        if self.last_goal_dist is not None:
            # goal_in_robot_frame : [rho, theta]

            # higher negative weight when moving away from goal
            # (to avoid driving unnecessary circles when train in contin. action space)
            if (self.last_goal_dist - goal_in_robot_frame[0]) > 0:
                w = reward_factor
            else:
                w = penalty_factor
            reward = w * (self.last_goal_dist - goal_in_robot_frame[0])

        self.last_goal_dist = goal_in_robot_frame[0]
        self._reward_composition["goal approached"] = reward
        return reward

    def _reward_collision(self, laser_scan: np.ndarray, punishment_factor: float = 10):
        """
        Reward for colliding with an obstacle.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for collision. defaults to 10
        """
        reward = 0
        if laser_scan.min() <= self.robot_radius + self.collision_tolerance:
            reward -= punishment_factor
            self.info["is_done"] = True
            self.info["done_reason"] = 1
            self.info["is_success"] = 0
            self.info["crash"] = True
        else:
            self.info["crash"] = False
        self._reward_composition["collision"] = reward
        return reward


    def _reward_safe_dist(self, laser_scan: np.ndarray, punishment_factor: float = 0.15):
        """
        Reward for undercutting safe distance.

        :param laser_scan (np.ndarray): laser scan data
        :param punishment (float, optional): punishment for undercutting. defaults to 0.15
        """
        reward = 0
        if laser_scan.min() < self.safe_dist:
            reward -= punishment_factor
            self.info["safe_dist"] = True
        self._reward_composition["safe dist"] = reward
        return reward

    def _reward_not_moving(self, action: np.ndarray = None, punishment_factor: float = 0.01):
        """
        Reward for not moving. Only applies half of the punishment amount
        when angular velocity is larger than zero.

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment for not moving. defaults to 0.01
        """
        reward = 0
        if action is not None and action[0] == 0.0:
            reward -= punishment_factor if action[1] == 0.0 else punishment_factor / 2
        self._reward_composition["not moving"] = reward
        return reward

    def _reward_distance_traveled(
        self,
        action: np.array = None,
        punishment: float = 0.01,
        consumption_factor: float = 0.005,
    ):
        """
        Reward for driving a certain distance. Supposed to represent "fuel consumption".

        :param action (np.ndarray (,2)): [0] - linear velocity, [1] - angular velocity
        :param punishment (float, optional): punishment when action can't be retrieved. defaults to 0.01
        :param consumption_factor (float, optional): weighted velocity punishment. defaults to 0.01
        """
        reward = 0
        if action is None:
            reward -= punishment
        else:
            lin_vel = action[0]
            ang_vel = action[1]
            reward += (lin_vel + (ang_vel * 0.001)) * consumption_factor
        self._reward_composition["distance traveled"] = reward
        return reward

    def _reward_distance_global_plan(
        self,
        global_plan: np.array,
        robot_pose: Pose2D,
        reward_factor: float = 0.1,
        penalty_factor: float = 0.15,
    ):
        """
        Reward for approaching/veering away the global plan. (Weighted difference between
        prior distance to global plan and current distance to global plan)

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param reward_factor (float, optional): positive factor when approaching global plan. defaults to 0.1
        :param penalty_factor (float, optional): negative factor when veering away from global plan. defaults to 0.15
        """
        reward = 0
        if global_plan is not None and len(global_plan) != 0:
            #print("calc plan dist reward")
            curr_dist_to_path, idx = self.get_min_dist2global_kdtree(
                global_plan, robot_pose
            )

            if self.last_dist_to_path is not None:
                if curr_dist_to_path < self.last_dist_to_path:
                    w = reward_factor
                else:
                    w = penalty_factor

                reward = w * (self.last_dist_to_path - curr_dist_to_path)
            self.last_dist_to_path = curr_dist_to_path
        #print(f"distance reward:{reward}")
        self._reward_composition["distance global plan"] = reward
        return reward

    def _reward_following_global_plan(
        self,
        global_plan: np.array,
        robot_pose: Pose2D,
        action: np.array = None,
        dist_to_path: float = 0.5,
        reward_factor: float = 0.1
    ):
        """
        Reward for travelling on the global plan.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        :param action (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        :param dist_to_path (float, optional): applies reward within this distance
        """
        reward = 0
        
        if global_plan is not None and len(global_plan) != 0 and action is not None:
            #print("calc plan follow reward")
            curr_dist_to_path, idx = self.get_min_dist2global_kdtree(
                global_plan, robot_pose
            )

            if curr_dist_to_path <= dist_to_path:
                reward = reward_factor * action[0]
        self._reward_composition["following global plan"] = reward
        #print(f"folowwing reward: {reward}")
        return reward

    def get_min_dist2global_kdtree(self, global_plan: np.array, robot_pose: Pose2D):
        """
        Calculates minimal distance to global plan using kd tree search.

        :param global_plan: (np.ndarray): vector containing poses on global plan
        :param robot_pose (Pose2D): robot position
        """
        if self.kdtree is None or self.last_global_plan is None or not np.array_equal(global_plan, self.last_global_plan):
            self.kdtree = scipy.spatial.cKDTree(global_plan)

        dist, index = self.kdtree.query([robot_pose.x, robot_pose.y])
        return dist, index

    def _reward_abrupt_direction_change(self, action: np.array = None, punishment_weight=2500):
        """
        Applies a penalty when an abrupt change of direction occured.

        :param action: (np.ndarray (,2)): [0] = linear velocity, [1] = angular velocity
        """
        reward = 0
        if self.last_action is not None:
            curr_ang_vel = action[1]
            last_ang_vel = self.last_action[1]

            vel_diff = abs(curr_ang_vel - last_ang_vel)
            reward = -(vel_diff ** 4) / punishment_weight
        self.last_action = action
        self._reward_composition["abrupt direction change"] = reward
        return reward