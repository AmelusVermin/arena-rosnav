#!/usr/bin/env python3

from distutils.command.config import config
from random import seed
import rospy
import time
import os
import rospkg
import sys
import subprocess
import signal
from pydoc import locate
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines.ppo2 import PPO2

from distutils.dir_util import copy_tree
from utils.startup_utils import load_vec_normalize
from utils.multi_threading_utils import set_pdeathsig
from utils.model_builder import ModelBuilder
from utils.argparser import get_config_arguments, get_commandline_arguments, check_params
from utils.startup_utils import make_envs, wait_for_nodes
from utils.custom_callbacks import StopTrainingOnRewardThreshold, InitiateNewTrainStage, EvalCallback
from policies.sb_policy_registry import PolicyRegistry
from utils.hyperparameter_utils import write_hyperparameters_json


def print_registered_types():
    all_types = PolicyRegistry.get_all_registered_agents()
    print("The following agent types are available:")
    for i, agent_type in enumerate(all_types):
        print(f"Type {i+1}: {agent_type}")

if __name__ == '__main__':
    # get command line arguments and params from relevant config files 
    args = get_commandline_arguments()
    # (environment_settings.yaml from this package and myrobot.model.yaml from simulator_setup)
    if args.configs_folder is "default":
        args.configs_folder = os.path.join(
            rospkg.RosPack().get_path("arena_combined_planner_drl"), 
            "configs",
            "configs"
        )

    settings_file = os.path.join(
            args.configs_folder, 
            "settings.yaml"
        )
    
    args, save_paths = get_config_arguments(args, settings_file)
    print(args.eval_freq)
    # -srt flag is set: print the registered agent types and exit
    if args.show_registered_types:
        print_registered_types()
        exit()
    
    # dict of existing log levels in ros
    
    # initiate ros node with according log level
    rospy.set_param("/enable_statistics", "true")
    rospy.set_param("/statistics_window_max_elements", 100)
    rospy.set_param("/statistics_window_min_elements", 10)
    rospy.set_param("/statistics_window_min_size", 4)
    rospy.set_param("/statistics_window_max_size", 10)
    rospy.set_param("/curr_stage", args.task_curr_stage)

    # for training with start_arena_flatland.launch
    ros_params = rospy.get_param_names()
    ns_for_nodes = "/single_env" not in ros_params
    ns_for_nodes = True
    # if no environment number was given on commandline, try to get it from ros
    if args.n_envs <= 0:
        try:
            # when starting flatland in training mode the number of envs was given, or the single_env as a param exist
            args.n_envs = 1 if not ns_for_nodes else rospy.get_param("num_envs")
        except KeyError:
            print("No environment number is given in ros! It is set to 1!")

            args.n_envs = 1
    
    # check and prepare some arguments
    check_params(args)
    args.n_steps = int(args.batch_size / args.n_envs)

    # wait for nodes to start
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=args.n_envs, timeout=5)

    # get classes of global and mid planner
    global_planner = locate(args.global_planner_class)
    mid_planner = locate(args.mid_planner_class)
    print(f"used global planner: {global_planner.get_name()}")
    print(f"used mid planner: {mid_planner.get_name()}")    
    # save configs to save dir
    copy_tree(args.configs_folder, os.path.join(save_paths['model'], "configs"))
    #copy_file(settings_file, save_paths['model'])
    #curriculum_file_name = save_paths['curriculum'].split("/")[-1]
    #copy_file(save_paths['curriculum'], os.path.join(save_paths['model'], curriculum_file_name))
    # for compatability issues the hyperparameters.json is neccessary (the task_generator uses it after an update)
    write_hyperparameters_json(args, save_paths)
    model_path = save_paths["model"]
    print(f"saving model data to: {model_path}")
    
    #unzip_map_parameters(save_paths, args.n_envs)
    # instantiate train environment
    # when debug run on one process only
    if ns_for_nodes:
        env = SubprocVecEnv(
            [
                make_envs(args, ns_for_nodes, i, global_planner, mid_planner, save_paths)
                for i in range(args.n_envs)
                
            ],
            start_method="fork"
        )
    else:
        env = DummyVecEnv(
            [
                make_envs(args, ns_for_nodes, i, global_planner, mid_planner, save_paths)
                for i in range(args.n_envs)
            ]
        )

    time.sleep((args.n_envs) * 13)
    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv(
            [make_envs(args, ns_for_nodes, 0, global_planner, mid_planner, save_paths, train=False)]
        )
    else:
        eval_env = env
        
    time.sleep(13)
    env, eval_env = load_vec_normalize(args, save_paths, env, eval_env)

    rospy.loginfo("create stop training callback")
    # stop training on reward threshold callback
    stoptraining_cb = StopTrainingOnRewardThreshold(
        treshhold_type=args.stop_threshhold_type, 
        threshold=args.stop_reward_threshhold, 
        verbose=args.stop_verbose
    )
    rospy.loginfo("create train stage callback")
    # threshold settings for training curriculum
    trainstage_cb = InitiateNewTrainStage(
        n_envs=args.n_envs,
        treshhold_type=args.stage_threshold_type,
        upper_threshold=args.stage_upper_thres,
        lower_threshold=args.stage_lower_thres,
        task_mode=args.task_mode,
        verbose=args.stage_verbose,
    )
    rospy.loginfo("create eval callback")
    # evaluation settings
    eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        log_path=save_paths['training'],
        model_save_path=save_paths['model'],
        deterministic=True,
        callback_on_eval_end=trainstage_cb,
        callback_on_new_best=stoptraining_cb,
    )

    # get model, either build new one or load a specified one
    model:PPO2 = ModelBuilder.get_model(args, save_paths, env)
    if args.overwrite_params:
        model.set_env(env)

    # start tensorboard
    command = ['tensorboard', f"--logdir={save_paths['tensorboard']}", "--port=6006"]
    tensorboard_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig(signal.SIGTERM))       
    
    # track training time
    rospy.loginfo("start training")
    start = time.time() 
    try:
        model.learn(
            total_timesteps=args.total_timesteps, callback=eval_cb, reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")

    if isinstance(env, VecNormalize):
        env.save(os.path.join(save_paths['model'], "vec_normalize.pkl"))
    model.save(os.path.join(save_paths['model'], "model"))          
    model.env.close()
    tensorboard_process.terminate()
    print(f"Time passed: {time.time()-start}s")
    print("Training script will be terminated")
    sys.exit()