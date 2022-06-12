import os
import random
import subprocess
import time
import signal
import rospkg
import rospy
import rosservice
import yaml
import traceback
from nav_msgs.srv import GetMap
from rospy import ServiceException
from task_generator.task_generator.obstacles_manager import ObstaclesManager
from task_generator.task_generator.robot_manager import RobotManager
from task_generator.task_generator.tasks import StagedRandomTask, get_predefined_task

from .multi_threading_utils import set_pdeathsig
from simulator_setup.srv import *


class TaskManager:

    def __init__(self, ns: str, paths: dict, run_scenario: bool, start_stage: int = 1, min_dist: float = 1, global_planner=None):
        self.ns = ns
        self.paths = paths
        self.run_scenario = run_scenario
        self.last_stage = 0
        self.min_dist = min_dist
        self.global_planner = global_planner
        if self.run_scenario:
            self.task = get_predefined_task(ns, mode='scenario', start_stage=1, min_dist=min_dist, PATHS=paths, global_planner=global_planner)
        else:
            self.task = self._get_random_task(paths, start_stage)
            self._request_new_map = rospy.ServiceProxy("/" + self.ns + "/new_map", GetMapWithSeed, persistent=True)


    def reset(self, seed, new_map=True):
        """ reset task and map """
        done = False
        attempts = 10
        while(not done):
            if not self.run_scenario:
                if new_map:
                    self._update_map(seed)
                random.seed(seed)
            try:
                self.task.reset()
                done = True
            except Exception:
                rospy.logwarn(f"{self.ns}: reset error, try again!")
                if attempts <= 0:
                    print(traceback.format_exc())
                    exit()
            attempts -= 1


    def _get_random_task(self, paths: dict, start_stage):
        """ prepare taskmanager """
        # load stage config
        config_path = paths['curriculum']
        with open(config_path, 'r') as params_json:
            map_params = yaml.safe_load(params_json)

        assert map_params is not None, "Error: training curriculum file cannot be found!"
        
        # get some params
        numb_static_obst = map_params[start_stage]['static']
        numb_dyn_obst = map_params[start_stage]['dynamic']
        map_type = map_params[start_stage]['map_type']
        if map_type == 'mixed':
            indoor_prob = map_params[start_stage]['indoor_prob']
        else:
            indoor_prob = 0

        #prepare map generator and service
        self._start_map_generator_node(map_type, indoor_prob)
        service_client_get_map = rospy.ServiceProxy('/' + self.ns + '/static_map', GetMap)

        service_name = '/' + self.ns + '/static_map'
        service_list = rosservice.get_service_list()
        max_tries = 10
        for i in range(max_tries):
            if service_name in service_list:
                break
            else:
                time.sleep(1)

        # crate Robot manager and Obstacle manager
        map_response = service_client_get_map()
        models_folder_path = rospkg.RosPack().get_path('simulator_setup')
        self.robot_manager = RobotManager(self.ns, map_response.map, os.path.join(
            models_folder_path, 'robot', "myrobot.model.yaml"), global_planner = self.global_planner)
        self.obstacles_manager = ObstaclesManager(self.ns, map_response.map)
        rospy.set_param("/task_mode", "staged")
        numb_obst = numb_static_obst + numb_dyn_obst
        if numb_obst != 0:
            prob_dyn_obst = float(numb_dyn_obst) / numb_obst
        else:
            prob_dyn_obst = 1
        self.obstacles_manager.register_random_obstacles(numb_obst, prob_dyn_obst)
        return StagedRandomTask(self.ns, self.obstacles_manager, self.robot_manager, start_stage, self.min_dist, self.paths)

    def _start_map_generator_node(self, map_type: str, indoor_prob: float):
        """ start map generator node """
        package = 'simulator_setup'
        launch_file = 'map_generator.launch'
        arg1 = "ns:=" + self.ns
        arg2 = "type:=" + map_type
        arg3 = "indoor_prob:=" + str(indoor_prob)
        command = ["roslaunch", package, launch_file, arg1, arg2, arg3]
        # Use subprocess to execute .launch file
        self._global_planner_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig(signal.SIGTERM))

    def _update_map_parameters(self, curr_stage):
        """ update map generator parameters """
        config_path = self.paths['curriculum']
        with open(config_path, 'r') as params_json:
            map_params = yaml.safe_load(params_json)

        assert map_params is not None, "Error: training curriculum file cannot be found!"
        
        # general map parameter
        rospy.set_param(f"/{self.ns}/map_generator_node/height", 
                        map_params[curr_stage]["map_height"])
        rospy.set_param(f"/{self.ns}/map_generator_node/width", 
                        map_params[curr_stage]["map_width"])
        rospy.set_param(f"/{self.ns}/map_generator_node/resolution", 
                        map_params[curr_stage]["map_resolution"])
        rospy.set_param(f"/{self.ns}/map_generator_node/map_type",
                        map_params[curr_stage]["map_type"])

        # indoor map parameter
        rospy.set_param(f"/{self.ns}/map_generator_node/corridor_radius", 
                        map_params[curr_stage]["indoor_corridor_radius"])
        rospy.set_param(f"/{self.ns}/map_generator_node/iterations", 
                        map_params[curr_stage]["indoor_iterations"])
        rospy.set_param(f"/{self.ns}/map_generator_node/room_number", 
                        map_params[curr_stage]["indoor_room_number"])
        rospy.set_param(f"/{self.ns}/map_generator_node/room_width", 
                        map_params[curr_stage]["indoor_room_width"])
        rospy.set_param(f"/{self.ns}/map_generator_node/room_height", 
                        map_params[curr_stage]["indoor_room_height"])
        rospy.set_param(f"/{self.ns}/map_generator_node/no_overlap", 
                        map_params[curr_stage]["indoor_room_no_overlap"])

        # outdoor map parameter
        rospy.set_param(f"/{self.ns}/map_generator_node/obstacle_number", 
                        map_params[curr_stage]["outdoor_obstacle_number"])
        rospy.set_param(f"/{self.ns}/map_generator_node/obstacle_extra_radius", 
                        map_params[curr_stage]["outdoor_obstacle_extra_radius"])

    def _update_map(self, seed: int):
        # update map parameters when stage has changed
        curr_stage = rospy.get_param("/curr_stage")
        while not isinstance(curr_stage, int) or curr_stage==0:
            curr_stage = rospy.get_param("/curr_stage")
            print(curr_stage)
        rospy.loginfo(f"{self.ns}: update map in stage {curr_stage}!")

        if self.last_stage != curr_stage or self.last_stage == 0:
            self._update_map_parameters(curr_stage)

        self.last_stage = curr_stage
        # request new map and update maanager
        request = GetMapWithSeedRequest(seed=seed)
        try:
            new_map = self._request_new_map(request)
            self.obstacles_manager.update_map(new_map.map)
            self.robot_manager.update_map(new_map.map)
            self.new_map = True
        except ServiceException:
            print(traceback.format_exc())
            exit()