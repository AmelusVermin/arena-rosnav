import rospy
import rospkg
import argparse
import yaml
import os

from .startup_utils import get_agent_name, setup_paths

LOG_LEVELS = {'debug' : rospy.DEBUG, 'info' : rospy.INFO, 'warn' : rospy.WARN, 'error' : rospy.ERROR}

def get_commandline_arguments():
    """parse the command line arguments with argparse"""
    # parse commandline arguments
    parser = argparse.ArgumentParser(description='Perform training of global, intermediate and local planner together.')
    parser.add_argument('--log_level', '-l', type=str, choices=['debug', 'info', 'warn', 'error'], default="info")
    parser.add_argument("--n_envs", '-ne', type=int, default=0, help="number of parallel environments.")
    parser.add_argument("--show_registered_types", '-srt', action="store_true", default=False, help="Prints the available agent types.")
    parser.add_argument("--configs_folder", '-cf', type=str, default="default", help="Loads the given settings files in the given folder instead of default one.")
    parser.add_argument("--eval_frequency", '-ef', type=int, default=None, help="replaces the eval_freq in settings.yaml")
    args = parser.parse_args()
    args.log_level = LOG_LEVELS[args.log_level]
    return args

def get_config_arguments(command_line_args: argparse.Namespace, settings_path:str):
    """parses all necessary configs, commandline arguments and prepares some arguments"""
    # get command line arguments
    args = command_line_args
    # load configs
    args = _add_robot_config(args)
    args = add_yaml_config(args, settings_path)
    if args.eval_frequency:
        args.eval_freq = args.eval_frequency
    # set agent name
    setattr(args, "agent_name", get_agent_name(args))
    # prepare some paths for logging and saving
    save_paths = setup_paths(args)
    return args, save_paths

def get_run_configs(config_path: str):
    args = argparse.Namespace()
    args = _add_robot_config(args)
    args = add_yaml_config(args, config_path)
    return args

def get_taks_manager_config(config_path: str):
    args = argparse.Namespace()
    args = add_yaml_config(args, config_path)
    return args

def _add_robot_config(args):
    """parses robot related config file from simulator setup package and adds important ones to the given args"""
    # get necessary robot parameters
    robot_settings_path = os.path.join(rospkg.RosPack().get_path("simulator_setup"), "robot","myrobot.model.yaml")
    
    # parse yaml and extract necessary data robot_radius, laser_num_beams, laser_max_range
    with open(robot_settings_path, 'r') as stream:
        try:
            robot_settings = yaml.safe_load(stream)
            # get robot radius
            for body in robot_settings["bodies"]:
                if body["name"] == "base_footprint":
                    for footprint in body["footprints"]:
                        if footprint["type"] == "circle":
                            setattr(
                                args, 
                                "robot_radius", 
                                (footprint.setdefault("radius", 0.3) * 1.05)
                                )
                        if footprint["radius"]:
                            setattr(
                                args, 
                                "robot_radius",
                                footprint["radius"] * 1.05
                                )
            # get laser related information
            for plugin in robot_settings["plugins"]:
                if plugin["type"] == "Laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    num_lidar_beams = int(
                        round((laser_angle_max - laser_angle_min) / laser_angle_increment) + 1
                    )
                    setattr(args, "num_lidar_beams", num_lidar_beams)
                    setattr(args, "lidar_range", plugin["range"])
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return args
    
def add_yaml_config(args: argparse.Namespace, config_path):
    """parses environment related config from config subfolder of this ros package"""
    with open(config_path, "r") as stream:
        try:
            # get robot related settings
            environment_settings = yaml.safe_load(stream)
            # flatten the yaml content to access more easily the nested params            
            flattened_dict = { k:v for k,v in _flatten_dict(environment_settings, "") }
            # add all params to args
            for k, v in flattened_dict.items():
                setattr(args, k, v)

        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return args

def _flatten_dict(pyobj, key):
    """ flattens a dict, assumes unique key names """
    if type(pyobj) == dict:
        for k in pyobj:
            yield from _flatten_dict(pyobj[k], str(k))
    else:
        yield key, pyobj

    

def check_params(args):
    assert (
        args.batch_size > args.mini_batch_size
    ), f"Mini batch size {args.mini_batch_size} is bigger than batch size {args.batch_size}"

    assert (
        args.batch_size % args.mini_batch_size == 0
    ), f"Batch size {args.batch_size} isn't divisible by mini batch size {args.mini_batch_size}"

    assert (
       args.batch_size % args.n_envs == 0
    ), f"Batch size {args.batch_size} isn't divisible by n_envs {args.n_envs}"

    assert (
       args.batch_size % args.mini_batch_size == 0
    ), f"Batch size {args.batch_size} isn't divisible by mini batch size {args.mini_batch_size}"
