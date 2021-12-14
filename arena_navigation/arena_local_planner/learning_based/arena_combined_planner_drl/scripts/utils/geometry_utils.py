import numpy as np
from geometry_msgs.msg import Pose2D
from tf.transformations import *

def get_pose_difference(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * np.pi) % (
            2 * np.pi
        ) - np.pi
        return rho, theta

def pose3D_to_pose2D(pose3d):
        """ convert 3D pose into 2D pose """
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

