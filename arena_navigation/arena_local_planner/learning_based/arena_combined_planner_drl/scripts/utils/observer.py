import numpy as np
from numpy.core.numeric import indices
import rospy
import message_filters
from typing import Tuple
from gym import spaces
from collections import deque
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from .geometry_utils import pose3D_to_pose2D, get_pose_difference

class Observer():

    def __init__(self, ns, args):
        # prepare namespace
        self.ns = ns
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"
  
        # define agent observation space, only for the agent itself!, it's used to define the input for the neural networks
        self._observation_space_type = args.observation_space_type
        self._num_global_plan_points = args.num_global_plan_points
        self._gp_point_skip_rate = args.gp_point_skip_rate
        self._num_lidar_beams = args.num_lidar_beams
        self._lidar_range = args.lidar_range
        self._max_distance_goal = args.max_distance_goal
        self.observation_space = self._prepare_observation_space(args)
        
        # define observation variables
        
        self._last_scan = None#LaserScan()
        self._scan_deque = deque(maxlen=args.max_deque_size)
        self._last_odom = None#Odometry()
        self._odom_deque = deque(maxlen=args.max_deque_size)
        self._last_goal = None#PoseStamped()
        
        # syncronization params
        self._sync_slop = args.sync_slop
        self._use_first_synced_obs = args.use_first_synced_obs

        # define subscribers
        # lidar scan
        self._scan_sub = message_filters.Subscriber(f"{self.ns_prefix}scan", LaserScan)
        # comment out if using TimeSynchronizer instead
        self._scan_sub.registerCallback(self._scan_callback)
        
        # odometry
        self._odom_sub = message_filters.Subscriber(f"{self.ns_prefix}odom", Odometry)
        # comment out if using TimeSynchronizer instead
        self._odom_sub.registerCallback(self._odom_callback)
        
        # define synchronizer for getting lidar scan and odometry at the same time
        # self._ts = message_filters.TimeSynchronizer([self._scan_sub, self._odom_sub], args.max_deque_size)
        # self._ts.registerCallback(self._sync_callback)
        
        # global goal
        self._goal_sub = message_filters.Subscriber(f"{self.ns_prefix}goal", PoseStamped)
        self._goal_sub.registerCallback(self._goal_callback)

    def _prepare_observation_space(self, args):
        
        # lidar scan
        scan = (spaces.Box(low=0, high=self._lidar_range, shape=(self._num_lidar_beams,), dtype=np.float32),)
        
        # used for any points as obs such as goal (distance and rotation to that point)
        point = (
            # distance to goal
            spaces.Box(low=0, high=self._max_distance_goal, shape=(1,), dtype=np.float32),
            # rotation angle to goal
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        )

        # length of the global path
        global_plan_length = (spaces.Box(low=0, high=self._max_distance_goal**2, shape=(1,), dtype=np.float32),)

        # given number of points from global path
        global_plan_path = point * self._num_global_plan_points

        if self._observation_space_type == "FULL_LENGTH":
            observation_space = Observer._stack_spaces(
                scan + point + point + global_plan_length
            )
        elif self._observation_space_type == "FULL_PATH":
            observation_space = Observer._stack_spaces(
                scan + point + point + global_plan_path
            )
        elif self._observation_space_type == "BASE":
            observation_space = Observer._stack_spaces(
                scan + point
            )
        elif self._observation_space_type == "SUBGOAL":
            observation_space = Observer._stack_spaces(
                scan + point + point
            )
        elif self._observation_space_type == "GLOBAL_LENGTH":
            observation_space = Observer._stack_spaces(
                scan + point + global_plan_length
            )
        elif self._observation_space_type == "GLOBAL_PATH":
            observation_space = Observer._stack_spaces(
                scan + point + global_plan_path
            )
        else:
            raise ValueError(f"observation_space: {self._observation_space_type} not known!")
        
        return observation_space

    def get_agent_observation_space(self):
        """ just returns the observation space for the agent only """
        return self.observation_space

    def _scan_callback(self, scan_msg):
        """ Callback for scan subscriber. Stores latest scan message in deque. """
        self._scan_deque.append(scan_msg)

    def _odom_callback(self, odom_msg):
        """ Callback for odom subscriber. Stores latest odom messag in deque. """
        self._odom_deque.append(odom_msg)

    def _sync_callback(self, scan, odom):
        """ callback for synchronized subscribers for lidar scan and odometry """
        self._last_scan = scan
        self._last_odom = odom

    def _goal_callback(self, msg_goal):
        """ callback for global goal subscriber """
        self._last_goal = msg_goal

    def get_observation(self):
        """get current observations from subscribed topics"""
        # get latest synced scan and odom message, not neccessary if TimeSynchronizer is used
        synced_scan, synced_odom = self.get_sync_obs()
        if synced_scan is not None and synced_odom is not None:
            self._last_scan = synced_scan
            self._last_odom = synced_odom
        # not all observations were set at least once in the current episode
        if self._last_odom is None or self._last_scan is None or self._last_goal is None:
            rospy.logdebug(f"odom: {self._last_odom}")
            rospy.logdebug(f"goal: {self._last_goal}")
            rospy.logdebug(f"scan: {self._last_scan}")
            return None
        # process messages
        scan = self.process_scan_msg(self._last_scan)
        robot_pose2D, twist = self.process_robot_state_msg(self._last_odom)
        # prepare output
        obs_dict = {
            "laser_scan": scan,
            "odom" : self._last_odom,
            "global_goal" : self._last_goal,
            "robot_pose": robot_pose2D,
            "twist" : twist
        }
        return obs_dict

    def get_sync_obs(self):
        """ slightly faster alternative to TimeSynchronizer for synchronized obtaining of scan and odometry messages. """
        synced_scan_msg = None
        synced_odom_msg = None
        rospy.logdebug(f"scan deque: {len(self._scan_deque)}, odom deque: {len(self._odom_deque)}")
        # iterate through deque of both topics
        while len(self._odom_deque) > 0 and len(self._scan_deque) > 0:
            # prepare first pair of messages to compare
            scan_msg = self._scan_deque.popleft()
            odom_msg = self._odom_deque.popleft()

            scan_stamp = scan_msg.header.stamp.to_sec()
            odom_stamp = odom_msg.header.stamp.to_sec()
            # try to find synced messages
            while abs(scan_stamp - odom_stamp) > self._sync_slop:
                rospy.logdebug(f"diff of stamps: {abs(scan_stamp - odom_stamp)}, sync_slop: {self._sync_slop}")
                # replace the older message of both by later one, return if end of deque reached
                if scan_stamp > odom_stamp:
                    if len(self._odom_deque) == 0:
                        return synced_scan_msg, synced_odom_msg
                    odom_msg = self._odom_deque.popleft()
                    odom_stamp = odom_msg.header.stamp.to_sec()
                else:
                    if len(self._scan_deque) == 0:
                        return synced_scan_msg, synced_odom_msg
                    scan_msg = self._scan_deque.popleft()
                    scan_stamp = scan_msg.header.stamp.to_sec()
            
            synced_scan_msg = scan_msg
            synced_odom_msg = odom_msg
            # if wished break the search if first synced pair found
            if self._use_first_synced_obs and synced_scan_msg is not None and synced_odom_msg is not None:
                break
        return synced_scan_msg, synced_odom_msg

    def prepare_agent_observation(self, robot_pose_2D, scan, goal, subgoal, global_plan_poses, global_plan_length):
        """ converts important information into the agent observation according to the observation space """
        # convert lidar scan to np array with appropriate data type
        if len(scan.ranges) > 0:
            scan_obs = scan.ranges.astype(np.float32)
        else:
            scan_obs = np.zeros(self._num_lidar_beams, dtype=float)
        # calculate difference between subgoal and robotpose to have an relative measurement for the agent
        goal_2D = pose3D_to_pose2D(goal)
        goal_obs = np.array(get_pose_difference(goal_2D, robot_pose_2D))

        # prepare subgoal observation
        if self._observation_space_type in ["FULL_LENGTH", "FULL_PATH", "SUBGOAL"]:
            subgoal_2D = pose3D_to_pose2D(subgoal)
            subgoal_obs = np.array(get_pose_difference(subgoal_2D, robot_pose_2D))

        # prepare global plan length observation
        if self._observation_space_type in ["FULL_LENGTH", "GLOBAL_LENGTH"]:
            length_obs = [global_plan_length]
        
        # prepare global plan path observation
        if self._observation_space_type in ["FULL_PATH", "GLOBAL_PATH"]:
            points = []
            rho, theta = 0.0, 0.0
            for i in range(0, self._num_global_plan_points * self._gp_point_skip_rate, self._gp_point_skip_rate):
                if i < len(global_plan_poses):
                    point_2D = pose3D_to_pose2D(global_plan_poses[i].pose)
                    rho, theta = get_pose_difference(point_2D, robot_pose_2D)
                points.append(rho)
                points.append(theta)
            gp_points_obs = np.array(points)

        # create agent observation according to observation_space_type
        if self._observation_space_type == "FULL_LENGTH":
            observation = np.hstack([scan_obs, goal_obs, subgoal_obs, length_obs])
        elif self._observation_space_type == "FULL_PATH":
            observation = np.hstack([scan_obs, goal_obs, subgoal_obs, gp_points_obs])
        elif self._observation_space_type == "BASE":
            observation = np.hstack([scan_obs, goal_obs])
        elif self._observation_space_type == "SUBGOAL":
            observation = np.hstack([scan_obs, goal_obs, subgoal_obs])
        elif self._observation_space_type == "GLOBAL_LENGTH":
            observation = np.hstack([scan_obs, goal_obs, length_obs])
        elif self._observation_space_type == "GLOBAL_PATH":
            observation = np.hstack([scan_obs, goal_obs, gp_points_obs])
        else:
            raise ValueError(f"observation_space: {self.observation_space} not known!")
        
        return observation
    
    

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
    
    def reset_deques(self):
        """ Clears both deques for synchronization. """
        self._scan_deque.clear()
        self._odom_deque.clear()
    
    def reset_last_obs(self):
        """ resets the member variables that contain last observations """
        self._last_goal = None
        self._last_odom = None
        self._last_scan = None

    def reset(self):
        """ resets observer for new episode """
        self.reset_deques()
        self.reset_last_obs()

    @staticmethod
    def process_global_plan_msg(globalplan):
        """ convert Path message to np array """
        global_plan_2d = list(
            map(
                lambda p: pose3D_to_pose2D(p.pose),
                globalplan.poses,
            )
        )
        return np.array(list(map(lambda p2d: [p2d.x, p2d.y], global_plan_2d)))

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low, dtype=np.float32).flatten(), np.array(high, dtype=np.float32).flatten())