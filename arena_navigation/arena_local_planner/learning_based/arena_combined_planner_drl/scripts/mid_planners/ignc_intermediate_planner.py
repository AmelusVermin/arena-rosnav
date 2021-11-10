from re import sub
import numpy as np
import rospy
from nav_msgs.msg import Odometry
import nav_msgs
import std_msgs
import rosservice
import rospkg
import os
import subprocess
import traceback
from geometry_msgs.msg import PoseStamped
from .mid_planner import MidPlanner
from intermediate_planner_interface.srv import MakeIntermediateGoal

NAME = "ignc_intermediate_planner"
class IntermediatePlanner(MidPlanner):

    def __init__(self, ns):
        super().__init__(ns)
        package = 'intermediate_planner_interface'
        launch_file = 'start_intermediate_planner_node.launch'
        ns = '""' if self.ns == "" else self.ns
        arg1 = f'ns:={self.ns}'
        arg2 = f"node_name:={IntermediatePlanner.get_name()}"
        command = ['roslaunch', package, launch_file, arg1, arg2]
        # start service
        self._global_planner_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)       

    @staticmethod
    def get_name() -> str:
        return NAME

    def get_subgoal(self, global_plan, odom):
        rospy.logdebug(f"waiting for mid planner service in namespace: '{self.ns}'")
        prefix = "" if self.ns == "" else f"/{self.ns}"
        rospy.wait_for_service(f'{prefix}/{IntermediatePlanner.get_name()}/makeSubgoal')
        make_subgoal = rospy.ServiceProxy(f'{prefix}/{IntermediatePlanner.get_name()}/makeSubgoal', MakeIntermediateGoal)
        rospy.logdebug(f"got intermediate planner service for namespace: '{self.ns}'")
        try:
            # prepare header, as service response doesn't have one
            header = header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = global_plan.header.frame_id
            # get response from service
            response = make_subgoal(global_plan, odom)
            # prepare output
            subgoal: PoseStamped = response.subgoal
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed in namespace '{self.ns}': {traceback.format_exc()}")

        return subgoal
