import gym
import numpy as np
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
        self.observation_space = Observer._stack_spaces(
            (
                # lidar scan
                spaces.Box(low=0, high=args.lidar_range, shape=(args.num_lidar_beams,), dtype=np.float32),
                # distance to (sub)goal
                spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
                # rotation angle to (sub)goal
                spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            )
        )

        # define observation variables
        self._num_lidar_beams = args.num_lidar_beams
        self._last_scan = LaserScan()
        self._scan_deque = deque(maxlen=args.max_deque_size)
        self._last_odom = Odometry()
        self._odom_deque = deque(maxlen=args.max_deque_size)
        self._last_goal = PoseStamped()
        
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
        return np.hstack([raw_scan, np.array([rho]), np.array([theta])])

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