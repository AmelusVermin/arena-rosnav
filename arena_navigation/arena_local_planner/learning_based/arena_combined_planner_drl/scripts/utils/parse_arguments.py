import rospy
import rospkg
import argparse
import yaml
import os

def get_arguments():
    """parses all necessary configs and commandline arguments"""
    args = _get_commandline_args()
    args = _add_robot_config(args)
    args = _add_environment_config(args)
    return args
    

def _get_commandline_args():
    """parse the command line arguments with argparse"""
    # parse commandline arguments
    parser = argparse.ArgumentParser(description='Perform training of global, intermediate and local planner together.')
    parser.add_argument('--log_level', '-l', type=str, choices=['debug', 'info', 'warn', 'error'], default="debug")
    return parser.parse_args()

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
                            args.robot_radius = (
                                footprint.setdefault("radius", 0.3) * 1.05
                            )
                        if footprint["radius"]:
                            args.robot_radius = footprint["radius"] * 1.05
            # get laser related information
            for plugin in robot_settings["plugins"]:
                if plugin["type"] == "Laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    args.num_lidar_beams = int(
                        round((laser_angle_max - laser_angle_min) / laser_angle_increment) + 1
                    )
                    args.lidar_range = plugin["range"]
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return args
    
def _add_environment_config(args):
    """parses environment related config from config subfolder of this ros package"""
    environment_settings_path = os.path.join(rospkg.RosPack().get_path("arena_combined_planner_drl"), "configs", "environment_settings.yaml")
    with open(environment_settings_path, "r") as stream:
        try:
            environment_settings = yaml.safe_load(stream)
            args.linear_range = environment_settings["robot"]["continuous_actions"]["linear_range"]
            args.angular_range = environment_settings["robot"]["continuous_actions"]["angular_range"]
            args.max_distance_goal = environment_settings["environment"]["max_distance_goal"]
            args.global_planner_class = environment_settings["environment"]["global_planner_class"]
            args.mid_planner_class = environment_settings["environment"]["mid_planner_class"]
            args.local_planner_class = environment_settings["environment"]["local_planner_class"]
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    return args

