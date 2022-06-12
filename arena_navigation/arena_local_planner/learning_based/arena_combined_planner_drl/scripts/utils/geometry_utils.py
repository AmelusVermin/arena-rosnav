import numpy as np
from geometry_msgs.msg import Pose2D
from tf.transformations import euler_from_quaternion

def get_pose_difference(pose1: Pose2D, pose2: Pose2D):
        assert not np.isnan([pose1.x, pose1.y, pose1.theta]).any(), f"pose1 has nan: {[pose1.x, pose1.y, pose1.theta]}" 
        assert not np.isnan([pose2.x, pose2.y, pose2.theta]).any(), f"pose1 has nan: {[pose2.x, pose2.y, pose2.theta]}" 
        y_relative = pose1.y - pose2.y
        x_relative = pose1.x - pose2.x
        rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
        theta = (np.arctan2(y_relative, x_relative) - pose2.theta + 4 * np.pi) % (
            2 * np.pi
        ) - np.pi
        assert not np.isnan([rho, theta]).any(), f"rho or theta is nan: {rho}, {theta}, {pose1}, {pose2}"
        assert not np.isinf([rho, theta]).any(), f"rho or theta is inf: {rho}, {theta}, {pose1}, {pose2}" 
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

def get_path_length(path: np.array):
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
