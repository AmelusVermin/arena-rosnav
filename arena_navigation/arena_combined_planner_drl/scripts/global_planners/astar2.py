import numpy as np
import rospy
import rospkg
from global_planners.global_planner import GlobalPlanner
from arena_services.srv import DummyService
from arena_path_search import dyn_astar

class AStar2(GlobalPlanner):

    def __init__(self):
        super().__init__()
        self.name = "astar2"

    def get_name(self):
        return self.name

    def plan_path(self, goal, odom):
        return np.array([odom, goal])
    
    def init(self):
        pass
        #rospy.wait_for_service('dummy_service')
        #try:
        #    dummy_service = rospy.ServiceProxy('dummy_service', DummyService)
        #    resp = dummy_service(1)
        #    rospy.loginfo(f"Service call: {resp}")
        #except rospy.ServiceException as e:
        #    rospy.logerror(f"Service call failed: {e}")
        