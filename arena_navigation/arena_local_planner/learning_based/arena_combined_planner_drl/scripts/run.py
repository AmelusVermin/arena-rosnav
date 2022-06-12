#! /usr/bin/env python3

import rospkg
import rospy
import message_filters
import os
import pickle
import time
import numpy as np

from queue import Empty
from time import sleep
from stable_baselines.ppo2 import PPO2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int16, Empty, Float32MultiArray
from utils.observer import Observer
from utils.argparser import get_run_configs
from utils.reward import RewardCalculator

class AgentNode():
    def __init__(self, args):
        # prepare namespace
        ns = rospy.get_param("ns", "")
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"
        self.deterministic = args.deterministic
        
        # instantiate observer        
        self._observer = Observer(ns, args, train_mode=False, goal_topic="goal")   
        
        # load model and pickle file
        self._model, self._n_env = self._load_model(args.model_path)
        self._normalize = self._load_pkl(args.vec_norm_path)
        self._reset = False
        
        # safe last recieved actions and goals
        self._last_goal = None
        self._last_action = None

        self.reward_calculator = RewardCalculator(
            robot_radius=args.robot_radius,
            safe_dist=args.robot_radius * 1.15,
            goal_radius=args.goal_radius,
            rule="rule_05",
            collision_tolerance=0.0,
            max_timesteps=2000
        )

        # prepare Publisher and Subscribers
        self._publish_rate = rospy.Rate(rospy.get_param("publish_rate", 20))
        self._action_publisher = rospy.Publisher(
            f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)

        self._goal_sub = message_filters.Subscriber(f"{self.ns_prefix}goal", PoseStamped, queue_size=1)
        self._goal_sub.registerCallback(self._goal_callback)

        self._scenario_reset = message_filters.Subscriber(f"/scenario_reset", Int16, queue_size=1)
        self._goal_sub.registerCallback(self._reset_callback)

        self._is_ready_pub = rospy.Publisher(f"{self.ns_prefix}is_ready", Empty, queue_size=1, latch=True)
        self._wait = True

        self._reward_pub = rospy.Publisher(f"{self.ns_prefix}reward", Float32MultiArray, queue_size=1, latch=True)
        self._response_time_pub = rospy.Publisher(f"{self.ns_prefix}response_time", Float32MultiArray, queue_size=1, latch=True)

    def _reset_callback(self, msg):
        """ reset callback """
        self._reset = True
        self._wait = True

    def _goal_callback(self, msg):
        """ If a new goal is set, reset observer """
        self._wait = False
        self._last_goal = msg

    def _load_pkl(self, vec_norm_file):
        """ loads the normalize function from pickle file if given """
        normalize_func = None
        if vec_norm_file:
            assert os.path.isfile(
                vec_norm_file
            ), f"VecNormalize file cannot be found at {vec_norm_file}!"

            with open(vec_norm_file, "rb") as file_handler:
                vec_normalize = pickle.load(file_handler)
            normalize_func = vec_normalize.normalize_obs
        return normalize_func

    def _load_model(self, model_zip):
        """ Loads the trained policy """
        
        assert os.path.isfile(
            model_zip
        ), f"Compressed model cannot be found at {model_zip}!"
        ppo_model = PPO2.load(model_zip)
        return ppo_model, ppo_model.n_envs
        
    def _prepare_action_msg(self, action):
        """ prepares twist message for action """
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        return action_msg

    def run_node(self):
        print("enter run loop")
        states = None
        done_masks = [False for _ in range(self._n_env)]
        self._is_ready_pub.publish(Empty())
        while not rospy.is_shutdown():
            
            # set masks for reset purpose
            if self._reset:
                done_masks = [True for _ in range(self._n_env)]
                states = None

            # get observations
            t1 = time.perf_counter()
            obs_dict = self._observer.get_deployment_observation()
            
            # check if observations are valid
            if obs_dict is not None:
                if any([v is None for v in obs_dict.values()]) or len(obs_dict["global_plan"].poses) <= 0:
                    print(f"some observations are None:")
                    for key, val in obs_dict.items():
                        if val is None: print(key)
                    sleep(0.2)
                    continue
                
                #process observations
                global_plan_array = Observer.process_global_plan_msg(obs_dict["global_plan"]) 
                proc_obs = self._observer.get_processed_observation(obs_dict, global_plan_array)
                
                # calculate rewards (for eval purposes)
                t2 = time.perf_counter()
                reward_follow_global = 0
                reward_dist_global = 0
                if self._last_action is not None:
                    reward_follow_global = self.reward_calculator._reward_following_global_plan(
                            global_plan_array, 
                            obs_dict["robot_pose"],
                            self._last_action,
                            dist_to_path=0.5, 
                            reward_factor=0.1
                    )
                if self._last_action is not None:
                    reward_dist_global = self.reward_calculator._reward_distance_global_plan(
                            global_plan_array, 
                            obs_dict["robot_pose"], 
                            reward_factor=0.2, 
                            penalty_factor=0.0
                    )

                # publish rewards for eval purposes
                msg = Float32MultiArray()
                msg.data = [reward_follow_global, reward_dist_global]
                self._reward_pub.publish(msg)
                t3 = time.perf_counter()

                # normalize observation of normalize function is given
                if self._normalize:
                    proc_obs = self._normalize(proc_obs)
                proc_obs = np.tile(proc_obs, (self._n_env, 1))
                
                # get action 
                actions, states = self._model.predict(proc_obs, states, done_masks, deterministic=self.deterministic)
                if self._reset:
                    done_masks = [False for _ in range(self._n_env)]
                    self._reset = False
                #publish action
                self._last_action = actions[0]
                action_msg = self._prepare_action_msg(actions[0])
                self._action_publisher.publish(action_msg)
                t4 = time.perf_counter()
                msg = Float32MultiArray()
                # (t3 - t2) is substracte as the reward calculation is not part of the actual deployment
                msg.data = [(t4 - t1) - (t3 - t2), t4-t3]
                self._response_time_pub.publish(msg)
                # sleep in given rate
            self._publish_rate.sleep()


if __name__ == "__main__":
    rospy.init_node(f"combined_planner_agent")
    config_path:str = rospy.get_param("/combined_planner_drl/config", "configs/run_configs/run_agent_3.yaml")
    pkg = rospkg.RosPack().get_path("arena_combined_planner_drl")

    #check if relative path is give and make it absolute if necessary
    if not config_path.startswith("/"):
        config_path = os.path.join(pkg, config_path)
    args = get_run_configs(config_path)

    #check if relative path is given and make it absolute if necessary
    if not args.model_path.startswith("/"):
        args.model_path = os.path.join(pkg, args.model_path)
    if not args.vec_norm_path.startswith("/"):
        args.vec_norm_path = os.path.join(pkg, args.vec_norm_path)


    agent_node = AgentNode(args)
    agent_node.run_node()