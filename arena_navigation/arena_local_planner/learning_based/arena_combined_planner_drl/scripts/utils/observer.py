import gym
import numpy as np
import rospy
import message_filters

from gym import spaces

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from utils.utils import pose3D_to_pose2D, get_pose_difference

class Observer():

    def __init__(self, ns, args):
        # prepare namespace
        self.ns = ns
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"

        # define agent observation space, only for the agent itself!, it's used to define the input for the neural networks
        self.observation_space = spaces.Tuple((
            # lidar scan
            spaces.Box(low=0, high=args.lidar_range, shape=(args.num_lidar_beams,), dtype=np.float32),
            # distance to (sub)goal
            spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            # rotation angle to (sub)goal
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        ))

        # define observation variables
        self._num_lidar_beams = args.num_lidar_beams
        self._last_scan = LaserScan()
        self._last_odom = Odometry()
        self._last_goal = PoseStamped()
        
        # define subscribers
        # lidar scan
        self._scan_sub = message_filters.Subscriber(f"{self.ns_prefix}/scan", LaserScan)
        # odometry
        self._odom_sub = message_filters.Subscriber(f"{self.ns_prefix}odom", Odometry)
        # define synchronizer for getting lidar scan and odometry at the same time
        self._ts = message_filters.TimeSynchronizer([self._scan_sub, self._odom_sub], 10)
        self._ts.registerCallback(self.sync_callback)
        # global goal
        self._goal_sub = message_filters.Subscriber(f"{self.ns_prefix}goal", PoseStamped)
        self._goal_sub.registerCallback(self.goal_callback)

    def get_agent_observation_space(self):
        """ just returns the observation space for the agent only """
        return self.observation_space

    def sync_callback(self, scan, odom):
        """ callback for synchronized subscribers for lidar scan and odometry """
        self._last_scan = scan
        self._last_odom = odom

    def goal_callback(self, msg_goal):
        """ callback for global goal subscriber """
        self._last_goal = msg_goal

    def get_observation(self):
        """get current observations from subscribed topics"""
        scan = self.process_scan_msg(self._last_scan)
        robot_pose2D, twist = self.process_robot_state_msg(self._last_odom)
        obs_dict = {
            "laser_scan": scan,
            "odom" : self._last_odom,
            "global_goal" : self._last_goal,
            "robot_pose": robot_pose2D,
            "twist" : twist
        }
        return obs_dict

    def prepare_agent_observation(self, scan, subgoal, robot_pose2D):
        """ converts important information into the agent observation according to the observation space """
        # convert lidar scan to np array with appropriate data type
        if len(scan.ranges) > 0:
            raw_scan = scan.ranges.astype(np.float32)
        else:
            raw_scan = np.zeros(self._num_lidar_beams, dtype=float)
        # calculate difference between subgoal and robotpose to have an relative measurement for the agent
        subgoal_2D = pose3D_to_pose2D(subgoal)
        rho, theta = get_pose_difference(subgoal_2D, robot_pose2D)
        # create observation according to observation space
        return tuple([raw_scan, np.array([rho]), np.array([theta])])

    def process_scan_msg(self, msg_LaserScan: LaserScan):
        """ remove_nans_from_scan """
        scan = np.array(msg_LaserScan.ranges)
        scan[np.isnan(scan)] = msg_LaserScan.range_max
        msg_LaserScan.ranges = scan
        return msg_LaserScan

    def process_robot_state_msg(self, msg_Odometry):
        """ split odometry message in 2D pose and twist """
        pose3d = msg_Odometry.pose.pose
        twist = msg_Odometry.twist.twist
        return pose3D_to_pose2D(pose3d), twist

    def process_pose_msg(self, msg_PoseWithCovarianceStamped):
        """ remove covariance """
        pose_with_cov = msg_PoseWithCovarianceStamped.pose
        pose = pose_with_cov.pose
        return pose3D_to_pose2D(pose)
    
