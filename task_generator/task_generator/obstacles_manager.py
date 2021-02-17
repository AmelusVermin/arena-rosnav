import math
import random
from typing import Union
import re
import yaml
import numpy as np
import os
import warnings
from flatland_msgs.srv import DeleteModel, DeleteModelRequest
from flatland_msgs.srv import SpawnModel, SpawnModelRequest
from flatland_msgs.srv import MoveModel, MoveModelRequest
from flatland_msgs.srv import StepWorld
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D, Twist, Point
from pedsim_srvs.srv import SpawnPeds
from pedsim_msgs.msg import Ped
from std_srvs.srv import SetBool, Empty
import rospy
import rospkg
import shutil
from utils import generate_freespace_indices, get_random_pos_on_map


class ObstaclesManager:
    """
    A manager class using flatland provided services to spawn, move and delete obstacles.
    """

    def __init__(self, ns: str, map_: OccupancyGrid):
        """
        Args:
            map_ (OccupancyGrid):
            plugin_name: The name of the plugin which is used to control the movement of the obstacles, Currently we use "RandomMove" for training and Tween2 for evaluation.
                The Plugin Tween2 can move the the obstacle along a trajectory which can be assigned by multiple waypoints with a constant velocity.Defaults to "RandomMove".
        """
        self.ns = ns
        self.ns_prefix = "/" if ns == '' else "/"+ns+"/"

        # a list of publisher to move the obstacle to the start pos.
        self._move_all_obstacles_start_pos_pubs = []

        # setup proxy to handle  services provided by flatland
        rospy.wait_for_service(f'{self.ns_prefix}move_model', timeout=20) 
        rospy.wait_for_service(f'{self.ns_prefix}delete_model', timeout=20)
        rospy.wait_for_service(f'{self.ns_prefix}spawn_model', timeout=20) 
        rospy.wait_for_service(f'{self.ns_prefix}pedsim_simulator/remove_all_peds', timeout=20)
        # start=rospy.rostime.get_time()
        # print("start wait ",start)
        rospy.wait_for_service(f'{self.ns_prefix}pedsim_simulator/respawn_peds' , timeout=20)
        rospy.wait_for_service(f'{self.ns_prefix}pedsim_simulator/spawn_ped' , timeout=20)
        # print("passed ",rospy.rostime.get_time()-start)
        is_training=rospy.get_param("/train_mode")
        if is_training:
            rospy.wait_for_service(f'{self.ns_prefix}step_world', timeout=20)
        # allow for persistent connections to services
        self._srv_move_model = rospy.ServiceProxy(
            f'{self.ns_prefix}move_model', MoveModel, persistent=True) 
        self._srv_delete_model = rospy.ServiceProxy(
            f'{self.ns_prefix}delete_model', DeleteModel, persistent=True) 
        self._srv_spawn_model = rospy.ServiceProxy(
            f'{self.ns_prefix}spawn_model', SpawnModel, persistent=True) 

        # self.__respawn_models = rospy.ServiceProxy('respawn_models', RespawnModels,persistent=True)
        # self.__respawn_models = rospy.ServiceProxy('%s/respawn_models' % self.NS, RespawnModels)
        self.__spawn_ped_srv = rospy.ServiceProxy(
            f'{self.ns_prefix}pedsim_simulator/spawn_ped', SpawnPeds, persistent=True)
        self.__respawn_peds_srv = rospy.ServiceProxy(
            f'{self.ns_prefix}pedsim_simulator/respawn_peds' , SpawnPeds, persistent=True)
        self.__remove_all_peds_srv = rospy.ServiceProxy(
            f'{self.ns_prefix}pedsim_simulator/remove_all_peds' , SetBool, persistent=True)
        # self._srv_sim_step = rospy.ServiceProxy('step_world', StepWorld, persistent=True)

        self.update_map(map_)
        self.static_obstacle_name_list = []
        self.__peds=[]
        self.obstacle_name_str=""
        self._obstacle_name_prefix = 'obstacle'
        # remove all existing obstacles generated before create an instance of this class
        self.remove_obstacles()
        # print("start wait ")
        # self.__remove_all_peds()
        # print("start wait ")
    def setForbidden_zones(self, forbidden_zones: Union[list, None] = None):
        """ set the forbidden areas for spawning obstacles
        """
        self.forbidden_zones=forbidden_zones 

    def update_map(self, new_map: OccupancyGrid):
        self.map = new_map
        # a tuple stores the indices of the non-occupied spaces. format ((y,....),(x,...)
        self._free_space_indices = generate_freespace_indices(self.map)

    def register_obstacles(self, num_obstacles: int, model_yaml_file_path: str, start_pos: list = [],type_obstacle:str='dynamic'):
        """register the obstacles defined by a yaml file and request flatland to respawn the them.

        Args:
            num_obstacles (string): the number of the obstacle instance to be created.
            model_yaml_file_path (string or None): model file path. it must be absolute path!
            start_pos (list)  a three-elementary list of empty list, if it is empty, the obstacle will be moved to the
                outside of the map.

        Raises:
            Exception:  Rospy.ServiceException(
                f" failed to register obstacles")

        Returns:
            self.
        """
        assert os.path.isabs(
            model_yaml_file_path), "The yaml file path must be absolute path, otherwise flatland can't find it"

        # the name of the model yaml file have the format {model_name}.model.yaml
        model_name = os.path.basename(model_yaml_file_path).split('.')[0]
        name_prefix = self._obstacle_name_prefix + '_' + model_name

        if type_obstacle == 'human':
            # print("asad",num_obstacles)
            self.__remove_all_peds()
            self.spawn_random_peds_in_world(num_obstacles)
        else:
            count_same_type = sum(
                1 if obstacle_name.startswith(name_prefix) else 0
                for obstacle_name in self.static_obstacle_name_list)

            for instance_idx in range(count_same_type, count_same_type + num_obstacles):
                max_num_try = 10
                i_curr_try = 0
                while i_curr_try < max_num_try:
                    spawn_request = SpawnModelRequest()
                    spawn_request.yaml_path = model_yaml_file_path
                    spawn_request.name = f'{name_prefix}_{instance_idx:02d}'
                    spawn_request.ns = rospy.get_namespace()
                    if len(start_pos) == 0:
                    # x, y, theta = get_random_pos_on_map(self._free_space_indices, self.map,)
                    # set the postion of the obstacle out of the map to hidden them
                        x = self.map.info.origin.position.x - 3 * \
                            self.map.info.resolution * self.map.info.height
                        y = self.map.info.origin.position.y - 3 * \
                            self.map.info.resolution * self.map.info.width
                        theta = theta = random.uniform(-math.pi, math.pi)
                    else:
                        assert len(start_pos) == 3
                        x = start_pos[0]
                        y = start_pos[1]
                        theta = start_pos[2]
                    spawn_request.pose.x = x
                    spawn_request.pose.y = y
                    spawn_request.pose.theta = theta
                    # try to call service
                    response = self._srv_spawn_model.call(spawn_request)
                    if not response.success:  # if service not succeeds, do something and redo service
                        rospy.logwarn(
                            f"spawn object {spawn_request.name} failed! trying again... [{i_curr_try+1}/{max_num_try} tried]")
                        rospy.logwarn(response.message)
                        i_curr_try += 1
                    else:
                        self.static_obstacle_name_list.append(spawn_request.name)
                        # self.obstacle_name_str=self.obstacle_name_str+","+spawn_request.name
                        break
                if i_curr_try == max_num_try:
                    raise rospy.ServiceException(f" failed to register static obstacles")        
        return self

    def register_random_obstacles(self, num_obstacles: int, p_dynamic=0.5):
        """register static or dynamic obstacles.

        Args:
            num_obstacles (int): number of the obstacles
            p_dynamic(float): the possibility of a obstacle is dynamic
            linear_velocity: the maximum linear velocity
        """
        num_dynamic_obstalces = int(num_obstacles*p_dynamic)
        max_linear_velocity = rospy.get_param("obs_vel")
        self.register_random_dynamic_obstacles(
            num_dynamic_obstalces, max_linear_velocity)
        self.register_random_static_obstacles(
            num_obstacles-num_dynamic_obstalces)
        rospy.loginfo(
            f"Registed {num_dynamic_obstalces} dynamic obstacles and {num_obstacles-num_dynamic_obstalces} static obstacles")

    def register_random_dynamic_obstacles(self, num_obstacles: int, linear_velocity=0.3, angular_velocity_max=math.pi/6, min_obstacle_radius=0.5, max_obstacle_radius=0.5):
        """register dynamic obstacles with circle shape.

        Args:
            num_obstacles (int): number of the obstacles.
            linear_velocity (float, optional):  the constant linear velocity of the dynamic obstacle.
            angular_velocity_max (float, optional): the maximum angular verlocity of the dynamic obstacle.
                When the obstacle's linear velocity is too low(because of the collision),we will apply an
                angular verlocity which is sampled from [-angular_velocity_max,angular_velocity_max] to the it to help it better escape from the "freezing" satuation.
            min_obstacle_radius (float, optional): the minimum radius of the obstacle. Defaults to 0.5.
            max_obstacle_radius (float, optional): the maximum radius of the obstacle. Defaults to 0.5.
        """
        # self.human_id=0
        # for i in range(num_obstacles):
        #     self.obstacle_name_str=self.obstacle_name_str+","+f'pedsim_agent_{i+1}/dynamic_human'
        model_path = os.path.join(rospkg.RosPack().get_path(
        'simulator_setup'), 'dynamic_obstacles/person_two_legged.model.yaml')
        self.register_obstacles(num_obstacles, model_path, type_obstacle='human')
            # os.remove(model_path)
##dynamic spawn in master branch
        # for _ in range(num_obstacles):
        #     model_path = self._generate_random_obstacle_yaml(
        #         True, linear_velocity=linear_velocity, angular_velocity_max=angular_velocity_max,
        #         min_obstacle_radius=min_obstacle_radius, max_obstacle_radius=max_obstacle_radius)
        #     self.register_obstacles(1, model_path)
        #     os.remove(model_path)

    def register_random_static_obstacles(self, num_obstacles: int, num_vertices_min=3, num_vertices_max=6, min_obstacle_radius=0.5, max_obstacle_radius=2):
        """register static obstacles with polygon shape.

        Args:
            num_obstacles (int): number of the obstacles.
            num_vertices_min (int, optional): the minimum number of the vertices . Defaults to 3.
            num_vertices_max (int, optional): the maximum number of the vertices. Defaults to 6.
            min_obstacle_radius (float, optional): the minimum radius of the obstacle. Defaults to 0.5.
            max_obstacle_radius (float, optional): the maximum radius of the obstacle. Defaults to 2.
        """
        for _ in range(num_obstacles):
            # print("reach here ")
            num_vertices = random.randint(num_vertices_min, num_vertices_max)
            model_path = self._generate_random_obstacle_yaml(
                False, num_vertices=num_vertices, min_obstacle_radius=min_obstacle_radius, max_obstacle_radius=max_obstacle_radius)
            self.register_obstacles(1, model_path, type_obstacle='static')
            os.remove(model_path)

    def register_static_obstacle_polygon(self, vertices: np.ndarray):
        """register static obstacle with polygon shape

        Args:
            verticies (np.ndarray): a two-dimensional numpy array, each row has two elements
        """
        assert vertices.ndim == 2 and vertices.shape[0] >= 3 and vertices.shape[1] == 2
        model_path, start_pos = self._generate_static_obstacle_polygon_yaml(
            vertices)
        self.register_obstacles(1, model_path, start_pos, type_obstacle='static')
        os.remove(model_path)

    def register_static_obstacle_circle(self,x,y,circle):
        model_path= self._generate_static_obstacle_circle_yaml(circle)
        self.register_obstacles(1, model_path, [x,y,0])
        os.remove(model_path)

    def register_dynamic_obstacle_circle_tween2(self, obstacle_name: str, obstacle_radius: float, linear_velocity: float, start_pos: Pose2D, waypoints: list, is_waypoint_relative: bool = True,  mode: str = "yoyo", trigger_zones: list = []):
        """register dynamic obstacle with circle shape. The trajectory of the obstacle is defined with the help of the plugin "tween2"

        Args:
            obstacle_name (str): the name of the obstacle
            obstacle_radius (float): The radius of the obstacle
            linear_velocity (float): The linear velocity of the obstacle
            start_pos (list): 3-elementary list
            waypoints (list): a list of 3-elementary list
            is_waypoint_relative (bool, optional): a flag to indicate whether the waypoint is relative to the start_pos or not . Defaults to True.
            mode (str, optional): [description]. Defaults to "yoyo".
            trigger_zones (list): a list of 3-elementary, every element (x,y,r) represent a circle zone with the center (x,y) and radius r. if its empty,
                then the dynamic obstacle will keeping moving once it is spawned. Defaults to True.
        """
        model_path, move_to_start_pub = self._generate_dynamic_obstacle_yaml_tween2(
            obstacle_name, obstacle_radius, linear_velocity, waypoints, is_waypoint_relative,  mode, trigger_zones)
        self._move_all_obstacles_start_pos_pubs.append(move_to_start_pub)
        self.register_obstacles(1, model_path, start_pos,type_obstacle='circle_random')
        os.remove(model_path)

    def move_all_obstacles_to_start_pos_tween2(self):
        for move_obstacle_start_pos_pub in self._move_all_obstacles_start_pos_pubs:
            move_obstacle_start_pos_pub.publish(Empty())
            print("moved ")

    def move_obstacle(self, obstacle_name: str, x: float, y: float, theta: float):
        """move the obstacle to a given position

        Args:
            obstacle_name (str): [description]
            x (float): [description]
            y (float): [description]
            theta (float): [description]
        """

        assert obstacle_name in self.static_obstacle_name_list, "can't move the obstacle because it has not spawned in the flatland"
        # call service move_model

        srv_request = MoveModelRequest()
        srv_request.name = obstacle_name
        srv_request.pose.x = x
        srv_request.pose.y = y
        srv_request.pose.theta = theta

        self._srv_move_model(srv_request)

    def reset_pos_obstacles_random(self, active_obstacle_rate: float = 1, forbidden_zones: Union[list, None] = None):
        """randomly set the position of all the obstacles. In order to dynamically control the number of the obstacles within the
        map while keep the efficiency. we can set the parameter active_obstacle_rate so that the obstacles non-active will moved to the
        outside of the map

        Args:
            active_obstacle_rate (float): a parameter change the number of the obstacles within the map
            forbidden_zones (list): a list of tuples with the format (x,y,r),where the the obstacles should not be reset.
        """
        active_obstacle_names = random.sample(self.static_obstacle_name_list, int(
            len(self.static_obstacle_name_list) * active_obstacle_rate))
        non_active_obstacle_names = set(
            self.static_obstacle_name_list) - set(active_obstacle_names)

        # non_active obstacles will be moved to outside of the map
        resolution = self.map.info.resolution
        pos_non_active_obstacle = Pose2D()
        pos_non_active_obstacle.x = self.map.info.origin.position.x - \
            resolution * self.map.info.width
        pos_non_active_obstacle.y = self.map.info.origin.position.y - \
            resolution * self.map.info.width

        for obstacle_name in active_obstacle_names:
            move_model_request = MoveModelRequest()
            move_model_request.name = obstacle_name
            # TODO 0.2 is the obstacle radius. it should be set automatically in future.
            move_model_request.pose.x, move_model_request.pose.y, move_model_request.pose.theta = get_random_pos_on_map(
                self._free_space_indices, self.map, 0.2, forbidden_zones)

            self._srv_move_model(move_model_request)

        for non_active_obstacle_name in non_active_obstacle_names:
            move_model_request = MoveModelRequest()
            move_model_request.name = non_active_obstacle_name
            move_model_request.pose = pos_non_active_obstacle
            self._srv_move_model(move_model_request)

    def _generate_dynamic_obstacle_yaml_tween2(self, obstacle_name: str, obstacle_radius: float, linear_velocity: float, waypoints: list, is_waypoint_relative: bool,  mode: str, trigger_zones: list):
        """generate a yaml file in which the movement of the obstacle is controller by the plugin tween2

        Args:
            obstacle_name (str): [description]
            obstacle_radius (float): [description]
            linear_velocity (float): [description]
            is_waypoint_relative (bool): [description]
            waypoints (list): [description]
            mode (str, optional): [description]. Defaults to "yoyo".
            trigger_zones (list): a list of 3-elementary, every element (x,y,r) represent a circle zone with the center (x,y) and radius r. if its empty,
                then the dynamic obstacle will keeping moving once it is spawned. Defaults to True.
        Returns:
            [type]: [description]
        """
        for i, way_point in enumerate(waypoints):
            if len(way_point) != 3:
                raise ValueError(
                    f"ways points must a list of 3-elementary list, However the {i}th way_point is {way_point}")
        tmp_folder_path = os.path.join(rospkg.RosPack().get_path(
            'simulator_setup'), 'tmp_random_obstacles')
        os.makedirs(tmp_folder_path, exist_ok=True)
        tmp_model_name = "dynamic_with_traj.model.yaml"
        yaml_path = os.path.join(tmp_folder_path, tmp_model_name)
        # define body
        body = {}
        body["name"] = "object_with_traj"
        body["type"] = "dynamic"
        body["color"] = [1, 0.2, 0.1, 1.0]
        body["footprints"] = []

        # define footprint
        f = {}
        f["density"] = 1
        f['restitution'] = 0
        f["layers"] = ["all"]
        f["collision"] = 'true'
        f["sensor"] = "false"
        # dynamic obstacles have the shape of circle
        f["type"] = "circle"
        f["radius"] = obstacle_radius

        body["footprints"].append(f)
        # define dict_file
        dict_file = {'bodies': [body], "plugins": []}
        # We added new plugin called RandomMove in the flatland repo
        move_with_traj = {}
        move_with_traj['type'] = 'Tween2'
        move_with_traj['name'] = 'Tween2 Plugin'
        move_with_traj['linear_velocity'] = linear_velocity
        # set the topic name for moving the object to the start point.
        # we can not use the flatland provided service to move the object, othewise the Tween2 will not work properly.
        move_with_traj['move_to_start_pos_topic'] = self.ns_prefix + obstacle_name + \
            '/move_to_start_pos'
        move_to_start_pos_pub = rospy.Publisher(
            move_with_traj['move_to_start_pos_topic'], Empty, queue_size=1)
        move_with_traj['waypoints'] = waypoints
        move_with_traj['is_waypoint_relative'] = is_waypoint_relative
        move_with_traj['mode'] = mode
        move_with_traj['body'] = 'object_with_traj'
        move_with_traj['trigger_zones'] = trigger_zones
        move_with_traj['robot_odom_topic'] = self.ns_prefix + 'odom'
        dict_file['plugins'].append(move_with_traj)

        with open(yaml_path, 'w') as fd:
            yaml.dump(dict_file, fd)
        return yaml_path, move_to_start_pos_pub

    def _generate_static_obstacle_polygon_yaml(self, vertices):
        # since flatland  can only config the model by parsing the yaml file, we need to create a file for every random obstacle
        tmp_folder_path = os.path.join(rospkg.RosPack().get_path(
            'simulator_setup'), 'tmp_random_obstacles')
        os.makedirs(tmp_folder_path, exist_ok=True)
        tmp_model_name = "polygon_static.model.yaml"
        yaml_path = os.path.join(tmp_folder_path, tmp_model_name)
        # define body
        body = {}
        body["name"] = "static_object"
        # calculate center of the obstacle
        obstacle_center = vertices.mean(axis=0)
        # convert to local coordinate system
        vertices = vertices - obstacle_center
        # convert to x,y,theta
        obstacle_center = obstacle_center.tolist()
        obstacle_center.append(0.0)

        body["type"] = "static"
        body["color"] = [0.2, 0.8, 0.2, 0.75]
        body["footprints"] = []

        # define footprint
        f = {}
        f["density"] = 1
        f['restitution'] = 0
        f["layers"] = ["all"]
        f["collision"] = 'true'
        f["sensor"] = "false"
        f["type"] = "polygon"
        f["points"] = vertices.astype(np.float).tolist()

        body["footprints"].append(f)
        # define dict_file
        dict_file = {'bodies': [body]}
        with open(yaml_path, 'w') as fd:
            yaml.dump(dict_file, fd)
        return yaml_path, obstacle_center

    def _generate_static_obstacle_circle_yaml(self,radius):
        # since flatland  can only config the model by parsing the yaml file, we need to create a file for every random obstacle
        tmp_folder_path = os.path.join(rospkg.RosPack().get_path(
            'simulator_setup'), 'tmp_random_obstacles')
        os.makedirs(tmp_folder_path, exist_ok=True)
        tmp_model_name = "circle_static.model.yaml"
        yaml_path = os.path.join(tmp_folder_path, tmp_model_name)
        # define body
        body = {}
        body["name"] = "static_object"
        body["type"] = "static"
        body["color"] = [0.2, 0.8, 0.2, 0.75]
        body["footprints"] = []

        # define footprint
        f = {}
        f["density"] = 1
        f['restitution'] = 0
        f["layers"] = ["all"]
        f["collision"] = 'true'
        f["sensor"] = "false"
        f["type"] = "circle"
        f["radius"] = radius

        body["footprints"].append(f)
        # define dict_file
        dict_file = {'bodies': [body]}
        with open(yaml_path, 'w') as fd:
            yaml.dump(dict_file, fd)
        return yaml_path

    def min_dist_check_passed(self,points):
        POINTS_MIN_DIST = 0.005*1.1
        points_1_x_2 = points[None,...]
        points_x_1_2 = points[:,None,:]
        points_dist = ((points_1_x_2-points_x_1_2)**2).sum(axis=2).squeeze()
        np.fill_diagonal(points_dist,1)
        min_dist = points_dist.min()
        return min_dist>POINTS_MIN_DIST

    def _generate_random_obstacle_yaml(self,
                                       is_dynamic=False,
                                       linear_velocity=0.3,
                                       angular_velocity_max=math.pi/4,
                                       num_vertices=3,
                                       min_obstacle_radius=0.5,
                                       max_obstacle_radius=1.5):
        """generate a yaml file describing the properties of the obstacle.
        The dynamic obstacles have the shape of circle,which moves with a constant linear velocity and angular_velocity_max

        and the static obstacles have the shape of polygon.

        Args:
            is_dynamic (bool, optional): a flag to indicate generate dynamic or static obstacle. Defaults to False.
            linear_velocity (float): the constant linear velocity of the dynamic obstacle. Defaults to 1.5.
            angular_velocity_max (float): the maximum angular velocity of the dynamic obstacle. Defaults to math.pi/4.
            num_vertices (int, optional): the number of vetices, only used when generate static obstacle . Defaults to 3.
            min_obstacle_radius (float, optional): Defaults to 0.5.
            max_obstacle_radius (float, optional): Defaults to 1.5.
        """

        # since flatland  can only config the model by parsing the yaml file, we need to create a file for every random obstacle
        # yaml_path
        tmp_folder_path = os.path.join(rospkg.RosPack().get_path(
            'simulator_setup'), 'tmp_random_obstacles')
        os.makedirs(tmp_folder_path, exist_ok=True)
        if is_dynamic:
            tmp_model_name = "random_dynamic.model.yaml"
        else:
            tmp_model_name = "random_static.model.yaml"
        yaml_path = os.path.join(tmp_folder_path, tmp_model_name)
        # define body
        body = {}
        body["name"] = "random"
        body["pose"] = [0, 0, 0]
        body["type"] = "static"
        body["color"] = [1, 0.2, 0.1, 1.0]  # [0.2, 0.8, 0.2, 0.75]
        body["footprints"] = []

        # define footprint
        f = {}
        f["density"] = 1
        f['restitution'] = 1
        f["layers"] = ["all"]
        f["collision"] = 'true'
        f["sensor"] = "false"
        # dynamic obstacles have the shape of circle
        f["type"] = "polygon"
        f["points"] = []
            # random_num_vert = random.randint(
            #     min_obstacle_vert, max_obstacle_vert)
        radius = random.uniform(
            min_obstacle_radius, max_obstacle_radius)
        # When we send the request to ask flatland server to respawn the object with polygon, it will do some checks
        # one important assert is that the minimum distance should be above this value
        # https://github.com/erincatto/box2d/blob/75496a0a1649f8ee6d2de6a6ab82ee2b2a909f42/include/box2d/b2_common.h#L65
        
        points = None
        while points is None:
            angles = 2*np.pi*np.random.random(num_vertices)
            points = np.array([np.cos(angles),np.sin(angles)]).T
            if not self.min_dist_check_passed(points):
                points = None
        f['points'] = points.tolist()

        body["footprints"].append(f)
        # define dict_file
        dict_file = {'bodies': [body], "plugins": []}
        # if is_dynamic:
        #     model_path = os.path.join(rospkg.RosPack().get_path(
        #     'simulator_setup'), 'dynamic_obstacles/person_two_legged.model.yaml')
        #     with open(model_path, 'r') as f:
        #         content = yaml.safe_load(f)
        #         # get laser_update_rate
        #         content['plugins'][1]['ground_truth_pub']=''
                    # if plugin['type'] == 'PosePub':
                        
            # We added new plugin called RandomMove in the flatland repo
        random_move = {}
        random_move['type'] = 'RandomMove'
        random_move['name'] = 'RandomMove Plugin'
        random_move['linear_velocity'] = linear_velocity
        random_move['angular_velocity_max'] = angular_velocity_max
        random_move['body'] = 'random'
        dict_file['plugins'].append(random_move)

        with open(yaml_path, 'w') as fd:
            yaml.dump(dict_file, fd)
        return yaml_path

    # def _get_dynamic_obstacle_configration(self, dynamic_obstacle_yaml_path):
    #     """get dynamic obstacle info e.g obstacle name, radius, Laser related infomation

    #     Args:
    #         dynamic_obstacle_yaml_path ([type]): [description]
    #     """
    #     # self.ROBOT_NAME = os.path.basename(dynamic_obstacle_yaml_path).split('.')[0]
    #     with open(dynamic_obstacle_yaml_path, 'r') as f:
    #         obstacle_data = yaml.safe_load(f)
    #         # get obstacle radius
    #         for body in obstacle_data['bodies']:
    #             if body['name'] == "base_footprint":
    #                 for footprint in body['footprints']:
    #                     if footprint['type'] == 'circle':
    #                         self.OBSTACLE_RADIUS = footprint.setdefault('radius', 0.2)


    def remove_obstacle(self, name: str):
        if len(self.static_obstacle_name_list) != 0:
            assert name in self.static_obstacle_name_list
        srv_request = DeleteModelRequest()
        srv_request.name = name
        response = self._srv_delete_model(srv_request)

        if not response.success:
            """
            raise rospy.ServiceException(
                f"failed to remove the object with the name: {name}! ")
            """
            warnings.warn(
                f"failed to remove the object with the name: {name}!")
        else:
            rospy.logdebug(f"Removed the obstacle with the name {name}")

    def remove_obstacles(self, prefix_names: Union[list, None] = None):
        """remove all the obstacless belong to specific groups.
        Args:
            prefix_names (Union[list,None], optional): a list of group names. if it is None then all obstacles will
                be deleted. Defaults to None.
        """
        if len(self.static_obstacle_name_list) != 0:
            if prefix_names is None:
                group_names = '.'
                re_pattern = "^(?:" + '|'.join(group_names) + r')\w*'
            else:
                re_pattern = "^(?:" + '|'.join(prefix_names) + r')\w*'
            r = re.compile(re_pattern)
            to_be_removed_obstacles_names = list(
                filter(r.match, self.static_obstacle_name_list))
            for n in to_be_removed_obstacles_names:
                self.remove_obstacle(n)
            self.static_obstacle_name_list = list(
                set(self.static_obstacle_name_list)-set(to_be_removed_obstacles_names))
        else:
            # it possible that in flatland there are still obstacles remaining when we create an instance of
            # this class.
            topics = rospy.get_published_topics()
            for t in topics:
                # the format of the topic is (topic_name,message_name)
                topic_name = t[0].split("/")
                object_name = topic_name[-1]
                if object_name.startswith(self._obstacle_name_prefix):
                    if "sim" in self.ns:
                        if self.ns in topic_name:
                            self.remove_obstacle(object_name)
                    else:
                        self.remove_obstacle(object_name)

    def __respawn_peds(self, peds):
        """
        Spawning one pedestrian in the simulation. The type of pedestrian is randomly decided here.
        TODO: the task generator later can decide the number of the agents
        ADULT = 0, CHILD = 1, ROBOT = 2, ELDER = 3,
        ADULT_AVOID_ROBOT = 10, ADULT_AVOID_ROBOT_REACTION_TIME = 11
        :param  start_pos start position of the pedestrian.
        :param  wps waypoints the pedestrian is supposed to walk to.
        :param  id id of the pedestrian.
        """
        srv = SpawnPeds()
        srv.peds = []
        # print(peds)
        self.agent_topic_str=''
        for i, ped in enumerate(peds):
            elements = [0, 1, 3]
            # probabilities = [0.4, 0.3, 0.3] np.random.choice(elements, 1, p=probabilities)[0]
            self.__ped_type=elements[i%3]
            if  self.__ped_type==0:
                self.agent_topic_str+=f',{self.ns_prefix}pedsim_agent_{i+1}/dynamic_human'
                self.__ped_file=os.path.join(rospkg.RosPack().get_path(
                'simulator_setup'), 'dynamic_obstacles/person_two_legged.model.yaml')
            elif self.__ped_type==1:
                self.agent_topic_str+=f',{self.ns_prefix}pedsim_agent_{i+1}/dynamic_child'
                self.__ped_file=os.path.join(rospkg.RosPack().get_path(
                'simulator_setup'), 'dynamic_obstacles/person_two_legged_child.model.yaml')
            else:
                # print('elder')
                self.agent_topic_str+=f',{self.ns_prefix}pedsim_agent_{i+1}/dynamic_elder'
                self.__ped_file=os.path.join(rospkg.RosPack().get_path(
                'simulator_setup'), 'dynamic_obstacles/person_single_circle_elder.model.yaml')
            msg = Ped()
            msg.id = ped[0]
            msg.pos = Point()
            msg.pos.x = ped[1][0]
            msg.pos.y = ped[1][1]
            msg.pos.z = ped[1][2]
            msg.type = self.__ped_type
            msg.number_of_peds = 1
            msg.yaml_file = self.__ped_file
            msg.waypoints = []
            for pos in ped[2]:
                p = Point()
                p.x = pos[0]
                p.y = pos[1]
                p.z = pos[2]
                msg.waypoints.append(p)
            srv.peds.append(msg)

        max_num_try = 2
        i_curr_try = 0
        while i_curr_try < max_num_try:
            # try to call service
            response=self.__respawn_peds_srv.call(srv.peds)
            # response=self.__spawn_ped_srv.call(srv.peds)
            if not response.finished:  # if service not succeeds, do something and redo service
                rospy.logwarn(
                    f"spawn human failed! trying again... [{i_curr_try+1}/{max_num_try} tried]")
                # rospy.logwarn(response.message)
                i_curr_try += 1
            else:
                break            
        # print("reached here respawn_humans")
        self.__peds = peds        
        rospy.set_param(f'{self.ns_prefix}agent_topic_string', self.agent_topic_str)
        return

    def spawn_random_peds_in_world(self, n:int, safe_distance:float=3.5, forbidden_zones: Union[list, None] = None):
        """
        Spawning n random pedestrians in the whole world.
        :param n number of pedestrians that will be spawned.
        :param map the occupancy grid of the current map (TODO: the map should be updated every spawn)
        :safe_distance [meter] for sake of not exceeding the safety distance at the beginning phase
        """
        ped_array = None
        # self.human_id+=1
        for i in range(n):
            waypoints = np.array([], dtype=np.int64).reshape(0, 3)
            [x, y, theta] = get_random_pos_on_map(self._free_space_indices, self.map, safe_distance, forbidden_zones)
            waypoints = np.vstack([waypoints, [x, y, 2]]) # the first waypoint
            # if random.uniform(0.0, 1.0) < 0.8:
            safe_distance=safe_distance-3 #the other waypoints don't need to avoid robot
            for j in range(4):
                dist = 0
                while dist < 4:
                    [x2, y2, theta2] = get_random_pos_on_map(self._free_space_indices, self.map, safe_distance, forbidden_zones)
                    dist = self.__mean_sqare_dist_((waypoints[-1,0] - x2), (waypoints[-1,1] - y2))
                waypoints = np.vstack([waypoints, [x2, y2, 2]])
            ped=np.array([i+1, [x, y, 0.0], waypoints])
            if ped_array==None:
                ped_array=ped
            else:
                ped_array=np.vstack([ped_array,ped]) 
        # print(ped_array.shape)
        self.__respawn_peds(ped_array)

    def __mean_sqare_dist_(self, x, y):
        """
        Computing mean square distance of x and y
        :param x, y
        :return: sqrt(x^2 + y^2)
        """
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

    def __remove_all_peds(self):
        """
        Removes all pedestrians, that has been spawned so far
        """
        srv = SetBool()
        srv.data = True
        # rospy.wait_for_service('%s/pedsim_simulator/remove_all_peds' % self.NS)
        self.__remove_all_peds_srv.call(srv.data)
        self.__peds = []
        return
