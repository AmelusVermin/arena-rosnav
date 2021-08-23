#!/usr/bin/env python
import rospy
import rospkg
import argparse
import yaml
import os
import torch as th
from environment import FlatlandEnv
from utils.parse_arguments import get_arguments
from pydoc import locate

def main():
    args = get_arguments()
    #dict of existing log levels in ros
    log_levels = {'debug' : rospy.DEBUG, 'info' : rospy.INFO, 'warn' : rospy.WARN, 'error' : rospy.ERROR}
    #initiate ros node with according log level
    rospy.init_node('trainer', log_level=log_levels[args.log_level])
    #debug output of parsed arguments
    rospy.logdebug(f"parsed arguments: {args}")

    global_planner = locate(args.global_planner_class)()
    global_planner.init()
    mid_planner = locate(args.mid_planner_class)()
    rospy.loginfo(f"used global planner: {global_planner.get_name()}")
    rospy.loginfo(f"used mid planner: {mid_planner.get_name()}")
    env = FlatlandEnv(args, global_planner, mid_planner)

    local_planner = locate(args.local_planner_class)(env.observation_space, 128)
    rospy.loginfo(f"used local planner: {local_planner.get_name()}")
    rospy.logdebug(f"local forward: {local_planner.forward(th.tensor([1,3]))}")
    
if __name__ == '__main__':
    main()