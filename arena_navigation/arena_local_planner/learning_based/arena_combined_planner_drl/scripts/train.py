#!/usr/bin/env python
import argparse
from warnings import catch_warnings
from arena_navigation.arena_local_planner.learning_based.arena_combined_planner_drl.scripts.utils.startup_utils import load_vec_normalize
import rospy
import time
import torch as th
import os
import rospkg
import subprocess
from typing import Type, Union
from pydoc import locate
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from distutils.file_util import copy_file
from utils.model_builder import ModelBuilder
from utils.argparser import get_arguments, check_params
from utils.startup_utils import make_envs, wait_for_nodes
from utils.staged_train_callback import InitiateNewTrainStage
from model.agent_factory import AgentFactory
from model.custom_policy import *
from model.simple_lstm_policy import *
from model.custom_sb3_policy import *

def print_registered_types():
    all_types = AgentFactory.get_all_registered_agents()
    print("The following agent types are available:")
    for i, agent_type in enumerate(all_types):
        print(f"Type {i+1}: {agent_type}")

def main():
    # get command line arguments and params from relevant config files 
    # (environment_settings.yaml from this package and myrobot.model.yaml from simulator_setup)
    settings_file = os.path.join(
        rospkg.RosPack().get_path("arena_combined_planner_drl"), 
        "configs", 
        "settings.yaml"
    )
    
    args, model_params, save_paths = get_arguments(settings_file)
    
    # remember used settings file
    if args.settings_file is not "":
        settings_file = args.settings_file
    
    # -srt flag is set: print the registered agent types and exit
    if args.show_registered_types:
        print_registered_types()
        exit()
    
    # dict of existing log levels in ros
    
    # initiate ros node with according log level
    rospy.set_param("/enable_statistics", "true")
    rospy.set_param("/statistics_window_max_elements", 100)
    rospy.set_param("/statistics_window_min_elements", 10)
    if args.debug:
        rospy.init_node('trainer', log_level=args.log_level)
    # debug output of parsed arguments
    rospy.logdebug(f"parsed arguments: {args}")

    #PATHS = get_paths("a1", args)
    # for training with start_arena_flatland.launch
    ros_params = rospy.get_param_names()
    ns_for_nodes = "/single_env" not in ros_params
    
    
    
    # if no environment number was given on commandline, try to get it from ros
    if args.n_envs <= 0:
        try:
            # when starting flatland in training mode the number of envs was given, or the single_env as a param exist
            args.n_envs = 1 if not ns_for_nodes else rospy.get_param("num_envs")
        except KeyError:
            rospy.logwarn("No environment number is given in ros! It is set to 1.")
            args.n_envs = 1
    
    check_params(args)
    args.n_steps = int(args.batch_size / args.n_envs)

    # wait for nodes to start
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=args.n_envs, timeout=5)

    # instantiate global and mid planner
    global_planner = locate(args.global_planner_class)()
    mid_planner = locate(args.mid_planner_class)()
    rospy.loginfo(f"used global planner: {global_planner.get_name()}")
    rospy.loginfo(f"used mid planner: {mid_planner.get_name()}")    
    
    # instantiate train environment
    # when debug run on one process only
    if not args.debug and ns_for_nodes:
        env = SubprocVecEnv(
            [
                make_envs(args, ns_for_nodes, i, global_planner, mid_planner, save_paths)
                for i in range(args.n_envs)
            ],
            start_method="fork",
        )
    else:
        env = DummyVecEnv(
            [
                make_envs(args, ns_for_nodes, i, global_planner, mid_planner, save_paths)
                for i in range(args.n_envs)
            ]
        )

    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv(
            [make_envs(args, ns_for_nodes, 0, global_planner, mid_planner, save_paths, train=False)]
        )
    else:
        eval_env = env
    
    env, eval_env = load_vec_normalize(args, save_paths, env, eval_env)

    # stop training on reward threshold callback
    stoptraining_cb = StopTrainingOnRewardThreshold(
        treshhold_type=args.stop_threshhold_type, threshold=args.stop_reward_threshhold, verbose=args.stop_verbose
    )

    # threshold settings for training curriculum
    trainstage_cb = InitiateNewTrainStage(
        n_envs=args.n_envs,
        treshhold_type=args.stage_threshold_type,
        upper_threshold=args.stage_upper_thres,
        lower_threshold=args.stage_lower_thres,
        task_mode=args.task_mode,
        verbose=args.stage_verbose,
    )

    # evaluation settings
    eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        log_path=save_paths['training'],
        best_model_save_path=save_paths['model'],
        deterministic=True,
        callback_on_eval_end=trainstage_cb,
        callback_on_new_best=stoptraining_cb,
    )

    

    # get model, either build new one or load a specified one
    model = ModelBuilder.get_model(args, model_params, save_paths, env)
    
    # save configs to save dir
    copy_file(settings_file, save_paths['model'])
    copy_file(args.model_config_path, save_paths['model'])
    curriculum_file_name = save_paths['curriculum'].split("/")[-1]
    copy_file(save_paths['curriculum'], os.path.join(save_paths['model'], curriculum_file_name))
    
    start = time.time() 
    try:
        model.learn(
            total_timesteps=args.total_timesteps, callback=eval_cb, reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")

    model.env.close()
    print(f"Time passed: {time.time()-start}s")
    print("Training script will be terminated")
    exit()

if __name__ == '__main__':
    main()