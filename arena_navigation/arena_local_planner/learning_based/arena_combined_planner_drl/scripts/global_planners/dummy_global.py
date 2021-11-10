import numpy as np

import std_msgs
from .global_planner import GlobalPlanner
import nav_msgs
import std_msgs.msg
import rospy
import geometry_msgs.msg


# name of this planner
NAME = "dummy global planner"

class Dummy(GlobalPlanner):

    def __init__(self, ns):
        super().__init__(ns)

    @staticmethod
    def get_name():
        """ Returns name of planner. """
        return NAME

    def get_global_plan(self, goal, odom):
        """ Creates a linear path from odom position to goal in 2 steps. """
        # prepare header for message

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = goal.header.frame_id

        #create path from current odom position to goal directly.

        global_path = nav_msgs.msg.Path()
        start = odom.pose.pose
        stamped_start = geometry_msgs.msg.PoseStamped()
        stamped_start.pose = start
        stamped_start.header = header
        global_path.poses.append(stamped_start)
        global_path.poses.append(goal)
        global_path.header = header
        return global_path

