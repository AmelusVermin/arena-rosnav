#! /usr/bin/env python3

from time import sleep

import rospkg
import rospy
import message_filters
import os
import pickle
from stable_baselines.ppo2 import PPO2
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int16
from utils.observer import Observer
from utils.argparser import get_run_configs
import time
import numpy as np

class AgentNode():
    def __init__(self, args):
        
        print("init run node")
        ns = rospy.get_param("ns", "")
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"

        self._observer = Observer(ns, args, train_mode=False, goal_topic="goal")   
        self._publish_rate = rospy.Rate(rospy.get_param("publish_rate", 20))
        self._action_publisher = rospy.Publisher(
            f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)
        self._model, self._n_env = self._load_model(args.model_path)
        self._normalize = self._load_pkl(args.vec_norm_path)
        self._reset = False
        # global goal
        self._last_goal = None
        self._goal_sub = message_filters.Subscriber(f"{self.ns_prefix}goal", PoseStamped, queue_size=1)
        self._goal_sub.registerCallback(self._goal_callback)

        self._scenario_reset = message_filters.Subscriber(f"/scenario_reset", Int16, queue_size=1)
        self._goal_sub.registerCallback(self._reset_callback)
        print("init done")
        

    def _reset_callback(self, msg):
        self._reset = True

    def _goal_callback(self, msg):
        """ If a new goal is set, reset observer """
        
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
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        return action_msg

    def run_node(self):
        print("enter loop")
        states = None
        done_masks = [False for _ in range(self._n_env)]
        while not rospy.is_shutdown():

            if self._reset:
                done_masks = [True for _ in range(self._n_env)]
                states = None

            # get observations
            print("get obs")
            obs_dict = self._observer.get_deployment_observation()
            if obs_dict is not None:
                if any([v is None for v in obs_dict.values()]):
                    print(f"some observations are None:")
                    for key, val in obs_dict.items():
                        if val is None: print(key)
                    sleep(0.2)
                    continue
                print("make plan array")
                global_plan_array = Observer.process_global_plan_msg(obs_dict["global_plan"])
                print("process obs")
                proc_obs = self._observer.get_processed_observation(obs_dict, global_plan_array)
                # normalize observation of normalize function is given
                print("normalize")
                if self._normalize:
                    proc_obs = self._normalize(proc_obs)
                proc_obs = np.tile(proc_obs, (self._n_env, 1))
                # get action 
                print(f"get predictions: {proc_obs.shape}")
                actions, states = self._model.predict(proc_obs, states, done_masks, deterministic=False)
                if self._reset:
                    done_masks = [False for _ in range(self._n_env)]
                    self._reset = False
                #publish action
                print("prepare action")
                action_msg = self._prepare_action_msg(actions[0])
                #print(action_msg)
                self._action_publisher.publish(action_msg)
                # sleep in given rate
            self._publish_rate.sleep()


if __name__ == "__main__":
    rospy.init_node(f"combined_planner_agent")
    config_path:str = rospy.get_param("/combined_planner_drl/config", "configs/default_configs/default_run_configs.yaml")
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