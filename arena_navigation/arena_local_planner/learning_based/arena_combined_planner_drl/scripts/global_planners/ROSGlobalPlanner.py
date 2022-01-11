from sys import stderr, stdout
import numpy as np
import rospy
import rospkg
import os
import subprocess
import time
from geometry_msgs.msg import Pose, PoseStamped
from rospy.exceptions import ROSException
from .global_planner import GlobalPlanner
from global_planner_interface.srv import MakeNewPlan
from utils.multi_threading_utils import set_pdeathsig
import nav_msgs
import std_msgs
import signal

NAME = "ROS_global_planner"
class ROSGlobalPlanner(GlobalPlanner):
    
    def __init__(self, ns):
        super().__init__(ns)

        # start planner service
        # prepare command 
        config_path = os.path.join(rospkg.RosPack().get_path('arena_combined_planner_drl'), 'configs', 'global_planner.yaml')
        package = 'global_planner_interface'
        launch_file = 'start_global_planner_node.launch'
        ns = '""' if self.ns == "" else self.ns
        arg1 = f'ns:={self.ns}'
        arg2 = f"node_name:={ROSGlobalPlanner.get_name()}"
        arg3 = f"config_path:={config_path}" 
        command = ['roslaunch', package, launch_file, arg1, arg2, arg3]
        # start service node
        self._global_planner_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig(signal.SIGTERM))       
        # prepare variables
        self.last_successful_plan = nav_msgs.msg.Path()

    def get_global_plan(self, goal, odom):
        """ Calls the global planner service with the given goal and returns response. Ignores odom """
        
        prefix = "" if self.ns == "" else f"/{self.ns}"
        make_new_plan = rospy.ServiceProxy(f'{prefix}/{ROSGlobalPlanner.get_name()}/makeGlobalPlan', MakeNewPlan)
        try:
            # prepare header, as service response doesn't have one
            header = header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = goal.header.frame_id
            # get response from service
            response = make_new_plan(goal)
            # prepare output
            global_plan: nav_msgs.msg.Path = response.global_plan
            global_plan.header = header
            rospy.logdebug(f"recieved path of length in namespace '{self.ns}': {len(global_plan.poses)}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed in namespace '{self.ns}': {e}")

        assert len(global_plan.poses) >= 2, "Global plan with only 1 or less poses was returned. This should not happen!"
        
        # check if emergency plan of start and end position was returned by service, this happen when no plan was found
        if len(global_plan.poses) == 2:
            if global_plan.poses[0].pose == global_plan.poses[1].pose:
                # replace global_plan by the last successful one
                global_plan == self.last_successful_plan
            else:
                self.last_successful_plan = global_plan

        return global_plan

    def _is_pose_equal(self, pose1 : PoseStamped, pose2 : PoseStamped):
        return pose1.pose == pose2.pose

    @staticmethod
    def get_name():
        return NAME

    def close(self):
        # stop service node
        self._global_planner_process.terminate()

    def is_ready(self):
        prefix = "" if self.ns == "" else f"/{self.ns}"
        try:
            rospy.logdebug(f"waiting for global planner service in namespace: '{self.ns}'")
            rospy.wait_for_service(f'{prefix}/{ROSGlobalPlanner.get_name()}/makeGlobalPlan', 0.25)
            rospy.logdebug(f"global planner service available for namespace: '{self.ns}'")
            return True
        except rospy.ROSException as e:
            return False