from re import A
import numpy as np
from geometry_msgs.msg import Pose2D
from tf.transformations import *
from scipy.interpolate import splprep, BSpline, splev, splrep
from scipy.integrate import trapz
import matplotlib.pyplot as plt

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

def get_path_length(path: np.array):
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))

def get_landmarks(global_plan_array):
    tck, u = get_bspline(global_plan_array)
    points = splev(u, tck, der=0)
    velocities = np.array(splev(u, tck, der=1)).transpose()
    accelerations = np.array(splev(u, tck, der=2)).transpose()
    acc_norm = np.cross(velocities, accelerations) / np.linalg.norm(velocities)    
    steering_angle_vel = np.array([np.linalg.norm(x) for x in acc_norm])  / np.array([np.linalg.norm(x) for x in velocities])
    steering_angle = trapz(steering_angle_vel)
    print(acc_norm)
    print(steering_angle_vel)
    print(steering_angle)
    return points
    

def get_bspline(global_plan_array):
    x = [p[0] for p in global_plan_array]
    y = [p[1] for p in global_plan_array]
    return splprep([x,y])

if __name__ == '__main__':
    a = [[2,1], [1,2], [2,3], [3,1], [4,0], [5,1], [6,4], [7,4.5]]
    ax = [p[0] for p in a]
    ay = [p[1] for p in a]
    
    b = get_landmarks(a)
    print(b)

    plt.plot(ax,ay, linestyle="solid")
    plt.plot(b[0],b[1], linestyle="dashed")
    plt.show()