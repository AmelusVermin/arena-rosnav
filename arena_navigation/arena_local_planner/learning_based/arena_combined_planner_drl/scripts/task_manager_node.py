#! /usr/bin/env python3
import rospy 
import rospkg
import os
import random
import subprocess
import signal
import std_msgs
import traceback
import time
import math

from utils.argparser import get_taks_manager_config
from utils.multi_threading_utils import set_pdeathsig
from task_generator.task_generator.tasks import get_predefined_task
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Int16
from rospy import ServiceException
from simulator_setup.srv import *
from nav_msgs.msg import Odometry

class TaskManagerNode:

    def __init__(self, args):
        # set seed
        self.seed = args.seed
        random.seed(self.seed)
        # prepare namespace
        self.ns = rospy.get_param("ns", "")
        self.ns_prefix = "" if (self.ns == ""  or self.ns is None) else f"/{self.ns}"

        # instantiate task and map generator
        self._task_manager = get_predefined_task(self.ns, 
                                mode='random_fixed', 
                                num_dyn_obstacles=args.dynamic, 
                                num_stat_obstacles=args.static, 
                                min_dist=args.min_dist, 
                                global_planner=None)
        self._map_generator = self._start_map_generator_node(args.map_type, 0.5)   
        time.sleep(10)
        self._request_new_map = rospy.ServiceProxy(self.ns_prefix + "/new_map", GetMapWithSeed, persistent=True)
        self._update_map_parameters(args)
        self._change_map = args.change_map 
        # prepare params nad 
        self.sr = rospy.Publisher('/scenario_reset', Int16, queue_size=1)
        self.nr = 0
        self.max_nr = rospy.get_param("~max_episodes", 10)


        # if auto_reset is set to true, the task generator will automatically reset the task
        # this can be activated only when the mode set to 'ScenarioTask'
        auto_reset = rospy.get_param("~auto_reset")

        # if the distance between the robot and goal_pos is smaller than this value, task will be reset
        self.timeout_= rospy.get_param("~timeout")
        self.timeout_= self.timeout_*60             # sec
        self.start_time_=time.time()                # sec
        self.delta_ = rospy.get_param("~delta")
        
      
        self.curr_goal_pos_ = None
        
        self._pause_sim = rospy.ServiceProxy(self.ns_prefix + "/pause", Empty)
        self._resume_sim = rospy.ServiceProxy(self.ns_prefix + "/resume", Empty)
        self._agent_ready_sub = rospy.Subscriber(f"{self.ns_prefix}/is_ready", std_msgs.msg.Empty, self._agent_callback, queue_size=1)
        self._is_agent_ready = False
        self.timer_is_locked = True
        if auto_reset:
            rospy.loginfo(
                "Task Generator is set to auto_reset mode, Task will be automatically reset as the robot approaching the goal_pos")
            self.reset_task()
            self.robot_pos_sub_ = rospy.Subscriber(
                f"{self.ns_prefix}/odom", Odometry, self.check_robot_pos_callback)
            rospy.Timer(rospy.Duration(0.5),self.goal_reached)
            
        else:
            # declare new service task_generator, request are handled in callback task generate
            self.reset_task()
        self._pause_sim()
        
        self.task_generator_srv_ = rospy.Service(
            'task_generator', Empty, self.reset_srv_callback)
                
        self.err_g = math.inf
        

    def _agent_callback(self, msg):
        """ callback for agent sends ready signal """
        self._is_agent_ready = True
        self.start_time_ = time.time()
        self._resume_sim()

    def goal_reached(self,event):
        """ Timer callback for checking timeout and goal reached status """
        if self._is_agent_ready and not self.timer_is_locked:
            if self.err_g < self.delta_:
                print(f"goal reached with dist: {self.err_g} smaller than threshold: {self.delta_}")
                self.reset_task()
            elif(time.time()-self.start_time_>self.timeout_):
                print("timeout")
                self.reset_task()

    def reset_srv_callback(self, req):
        """ reset service callback """
        rospy.loginfo("Task Generator received task-reset request!")
        self.reset_task()
        return EmptyResponse()

    def reset_task(self):
        """ resets task and map """
        self.timer_is_locked = True
        print(f"reset map and task: epsisode {self.nr}")
        if self.nr < self.max_nr:
            if not self._change_map:
                random.seed(self.seed)
            self._update_map()
            self._reset_costmap()
            time.sleep(0.3)
            info = self._task_manager.reset()
            
            if info is not None:
                self.curr_goal_pos_ = info['robot_goal_pos']
            rospy.loginfo("".join(["="]*80))
            rospy.loginfo("goal reached and task reset!")
            rospy.loginfo("".join(["="]*80)
            )
            
            self.start_time_=time.time()
            self.err_g = math.inf
            self.sr.publish(self.nr)
            self.nr += 1
        self.timer_is_locked = False

    def _reset_costmap(self):
        """ reset costmap """
        reset_costmap = rospy.ServiceProxy(f'{self.ns_prefix}/move_base/clear_costmaps', Empty)
        try:
            reset_costmap()
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed in namespace '{self.ns}': {e}")

    def check_robot_pos_callback(self, odom_msg: Odometry):
        """ calculate distance from odom to goal """
        robot_pos = odom_msg.pose.pose.position
        robot_x = robot_pos.x
        robot_y = robot_pos.y
        if self._task_manager.robot_manager._curr_goal is not None:
            goal_x = self._task_manager.robot_manager._curr_goal.x
            goal_y = self._task_manager.robot_manager._curr_goal.y

            self.err_g = (robot_x-goal_x)**2+(robot_y-goal_y)**2
        else:
            self.err_g = math.inf

    def _update_map(self):
        """ updates map """
        if self._change_map:
            seed = self.nr + self.seed
        else:
            seed = self.seed
        request = GetMapWithSeedRequest(seed=seed)
        try:
            new_map = self._request_new_map(request)
            self._task_manager.obstacles_manager.update_map(new_map.map)
            self._task_manager.robot_manager.update_map(new_map.map)
            self.new_map = True
        except ServiceException:
            print(traceback.format_exc())
            exit()
    
    def _update_map_parameters(self, args):
        """ set map parameters according to config values """
        # general map parameter
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/height", 
                        args.map_height)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/width", 
                        args.map_width)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/resolution", 
                        args.map_resolution)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/map_type",
                        args.map_type)

        # indoor map parameter
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/corridor_radius", 
                        args.indoor_corridor_radius)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/iterations", 
                        args.indoor_iterations)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/room_number", 
                        args.indoor_room_number)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/room_width", 
                        args.indoor_room_width)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/room_height", 
                        args.indoor_room_height)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/no_overlap", 
                        args.indoor_room_no_overlap)

        # outdoor map parameter
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/obstacle_number", 
                        args.outdoor_obstacle_number)
        rospy.set_param(f"{self.ns_prefix}/map_generator_node/obstacle_extra_radius", 
                        args.outdoor_obstacle_extra_radius)

    def _start_map_generator_node(self, map_type: str, indoor_prob: float):
        """ start map generator node """
        package = 'simulator_setup'
        launch_file = 'map_generator.launch'
        arg1 = "ns:=" + self.ns if self.ns != "" or self.ns is not None else "ns:=''"
        arg2 = "type:=" + map_type
        arg3 = "indoor_prob:=" + str(indoor_prob)
        command = ["roslaunch", package, launch_file, arg1, arg2, arg3]
        # Use subprocess to execute .launch file
        print(f"start node with:{command}")
        self._global_planner_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig(signal.SIGTERM))

if __name__ == "__main__":
    rospy.init_node(f"task_manager_node")
    config_path:str = rospy.get_param("~config", "configs/default_configs/default_task_manager_configs.yaml")
    pkg = rospkg.RosPack().get_path("arena_combined_planner_drl")

    #check if relative path is give and make it absolute if necessary
    if not config_path.startswith("/"):
        config_path = os.path.join(pkg, config_path)
    args = get_taks_manager_config(config_path)

    node = TaskManagerNode(args)
    rospy.spin()
