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
from global_planner_interface.srv import MakeGlobalPlan, MakeGlobalPlanFull, ResetCostmap
from utils.multi_threading_utils import set_pdeathsig
import nav_msgs
import std_msgs
import signal

NAME = "ROS_global_planner"
class ROSGlobalPlanner(GlobalPlanner):
    
    def __init__(self, ns:str, config_folder:str):
        super().__init__(ns)

        #start planner service
        #prepare command 
        config_file = os.path.join(config_folder, 'global_planner.yaml')
        package = 'global_planner_interface'
        launch_file = 'start_global_planner_node.launch'
        ns = '""' if self.ns == "" else self.ns
        arg1 = f'ns:={ns}'
        arg2 = f"node_name:={ROSGlobalPlanner.get_name()}"
        arg3 = f"config_path:={config_file}" 
        command = ['roslaunch', package, launch_file, arg1, arg2, arg3]
        # start service node
        self._global_planner_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig(signal.SIGTERM))       
        # prepare variables
        self._last_successful_plan = None

    def get_global_plan(self, goal, odom):
        """ Calls the global planner service with the given goal and returns response. Ignores odom """
        
        try:
            # prepare header, as service response doesn't have one
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = goal.header.frame_id
            # get response from service
            start = PoseStamped()
            start.header = header
            start.pose = odom.pose.pose
            response = self._make_new_plan(start, goal)
            # prepare output
            global_plan: nav_msgs.msg.Path = response.global_plan
            success: bool = response.success
            global_plan.header = header
            rospy.logdebug(f"recieved path of length in namespace '{self.ns}': {len(global_plan.poses)}, {success}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed in namespace '{self.ns}': {e}")

        # check if plan with 2 poses was returned, this happens when no plan was found or the robot is pretty close to the goal
        if not success:
            # replace global_plan by the last successful one
            if self._last_successful_plan is None:
                global_plan.poses.append(start)
                global_plan.poses.append(goal)
            else:
                global_plan = self._last_successful_plan
        else:
            self._last_successful_plan = global_plan
        
        return global_plan, success

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
            prefix = "" if self.ns == "" else f"/{self.ns}"
            self._make_new_plan = rospy.ServiceProxy(f'{prefix}/{ROSGlobalPlanner.get_name()}/makeGlobalPlanFull', MakeGlobalPlanFull)
            #self._make_new_plan.queue_size = 1
            return True
        except rospy.ROSException as e:
            return False

    def reset(self):
        self._last_successful_plan = None
        prefix = "" if self.ns == "" else f"/{self.ns}"
        reset_costmap = rospy.ServiceProxy(f'{prefix}/{ROSGlobalPlanner.get_name()}/resetCostmap', ResetCostmap)
        try:
            reset_costmap()
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed in namespace '{self.ns}': {e}")
