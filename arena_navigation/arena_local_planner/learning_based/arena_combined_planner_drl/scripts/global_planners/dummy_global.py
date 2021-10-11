import numpy as np

import std_msgs
from .global_planner import GlobalPlanner
import nav_msgs
import std_msgs.msg
import rospy
import geometry_msgs.msg

class Dummy(GlobalPlanner):

    def __init__(self):
        super().__init__()
        self.name = "dummy global planner"

    def get_name(self):
        return self.name

    def get_global_plan(self, goal, odom):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = goal.header.frame_id

        global_path = nav_msgs.msg.Path()
        start = odom.pose.pose
        stamped_start = geometry_msgs.msg.PoseStamped()
        stamped_start.pose = start
        stamped_start.header = header
        global_path.poses.append(stamped_start)
        global_path.poses.append(goal)
        global_path.header = header
        return global_path

