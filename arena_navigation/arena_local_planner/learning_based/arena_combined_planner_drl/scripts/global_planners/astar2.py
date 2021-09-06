import numpy as np
import rospy
import rospkg
import os
import subprocess
import rosservice
from geometry_msgs.msg import Pose, PoseStamped
from global_planners.global_planner import GlobalPlanner
from global_planner_interface.srv import MakeNewPlan

class AStar2(GlobalPlanner):

    def __init__(self, ns):
        super().__init__()
        self.name = "global_planner"
        self.ns = ns
        self._global_planner_process = None

    def get_name(self):
        return self.name

    def plan_path(self, goal, odom):
        return np.array([odom, goal])
    
    def init(self):
        self._start_global_planner()
        print(rosservice.get_service_list())
        rospy.wait_for_service('makeGlobalPlan')
        try:
            make_new_plan = rospy.ServiceProxy('makeGlobalPlan', MakeNewPlan)
            goal_msg = PoseStamped()
            goal_pos = Pose()
            goal_pos.position.x = 20
            goal_pos.position.y = 10
            goal_pos.position.z = 0
            print(goal_pos.orientation.x, goal_pos.orientation.w)
            goal.header.frame_id = "map"
            resp = make_new_plan(1)
            rospy.loginfo(f"Service call: {resp}")
        except rospy.ServiceException as e:
            rospy.logerror(f"Service call failed: {e}")

    def _start_global_planner(self):
        config_path = os.path.join(rospkg.RosPack().get_path('arena_combined_planner_drl'), 'configs', 'global_planner.yaml')
        package = 'global_planner_interface'
        launch_file = 'start_global_planner_node.launch'
        arg1 = f"ns:={self.ns}"
        arg2 = f"node_name:={self.name}"
        arg3 = f"config_path:={config_path}" 
        self._global_planner_process = subprocess.Popen(["roslaunch", package, launch_file, arg1, arg2, arg3], stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
