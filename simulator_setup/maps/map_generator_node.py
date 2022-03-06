#!/usr/bin/env python3

import random
import numpy as np
import nav_msgs.srv
import rospkg
import rospy
import os
from map_generator import create_yaml_files, create_random_map, make_image 
from nav_msgs.msg import OccupancyGrid

from simulator_setup.srv import *

class MapGenerator:
    def __init__(self):
        # initial value for scenario number
        self.nr = -1

        self._update_params()

        # initialize occupancy grid
        self.occupancy_grid = OccupancyGrid()
        self.ns = rospy.get_param("~ns")
        
        self.indoor_prob = rospy.get_param("~indoor_prob")
        
        # self.generate_initial_map() # initial random map generation (before first episode)
        rospy.Subscriber("/" + self.ns + '/map', OccupancyGrid, self.get_occupancy_grid)
        # generate new random map for the next episode when entering new episode
        rospy.Service("/" + self.ns + '/new_map', GetMapWithSeed, self.new_episode_callback)
        self.mappub = rospy.Publisher("/" + self.ns + '/map', OccupancyGrid, queue_size=1)

        # initialize yaml files
        map_dir = os.path.join(rospkg.RosPack().get_path('simulator_setup'), 'maps')
        create_yaml_files('random_map', map_dir, self.resolution, self.ns)

    # a bit cheating: copy OccupancyGrid meta data from map_server of initial map
    def get_occupancy_grid(self, occgrid_msg):
        self.occupancy_grid = occgrid_msg

    def generate_initial_map(self):  # generate random map png in random_map directory
        map = create_random_map(
            height=self.height,
            width=self.width,
            corridor_radius=self.corridor_radius,
            iterations=self.iterations,
            obstacle_number=self.obstacle_number,
            obstacle_extra_radius=self.obstacle_extra_radius,
            room_number=self.room_number,
            room_width=self.room_width,
            room_height=self.room_height,
            no_overlap=self.no_overlap,
            map_type=self.map_type,
            indoor_prob=self.indoor_prob,
            seed=0
        )
        make_image(map, self.ns)
        rospy.loginfo("Initial random map generated.")

    def generate_mapdata(self, seed: int = 0):  
        # generate random map data array for occupancy grid
        map = create_random_map(
            height=self.height,
            width=self.width,
            corridor_radius=self.corridor_radius,
            iterations=self.iterations,
            obstacle_number=self.obstacle_number,
            obstacle_extra_radius=self.obstacle_extra_radius,
            room_number=self.room_number,
            room_width=self.room_width,
            room_height=self.room_height,
            no_overlap=self.no_overlap,
            map_type=self.map_type,
            indoor_prob=self.indoor_prob,
            seed=seed
        )
        make_image(map, self.ns)
        map = np.flip(map, axis=0)
        # map currently [0,1] 2D np array needs to be flattened for publishing OccupancyGrid.data
        map = (map * 100).flatten()
        return map

    # def new_episode_callback(self,goal_msg: PoseStamped):
    #     current_episode = goal_msg.header.seq
    #     is_new_episode = self.nr != current_episode # self.nr starts with -1 so 0 will be the first new episode
    #     if is_new_episode:
    #         self.nr = current_episode
    #         self.occupancy_grid.data = self.generate_mapdata()
    #         rospy.loginfo("New random map generated for episode {}.".format(self.nr))
    #         self.mappub.publish(self.occupancy_grid)
    #         rospy.loginfo("New random map published.")

    def new_episode_callback(self, request: GetMapWithSeedRequest):
        seed = request.seed
        self._update_params()
        occ_grid = self.occupancy_grid
        occ_grid.data = self.generate_mapdata(seed)
        occ_grid.info.height = self.height
        occ_grid.info.width = self.width
        occ_grid.info.resolution = self.resolution
        self.mappub.publish(occ_grid)
        self.occupancy_grid = occ_grid
        srv_response = GetMapWithSeedResponse(map=occ_grid)
        return srv_response

    def _update_params(self):
        # general map parameter
        self.height = rospy.get_param("~height")
        self.width = rospy.get_param("~width")
        self.resolution = rospy.get_param("~resolution")
        self.map_type = rospy.get_param("~map_type")
        # indoor map parameter
        self.corridor_radius = rospy.get_param("~corridor_radius")
        self.iterations = rospy.get_param("~iterations")
        self.room_number = rospy.get_param("~room_number")
        self.room_width = rospy.get_param("~room_width")
        self.room_height = rospy.get_param("~room_height")
        self.no_overlap = rospy.get_param("~no_overlap")

        # outdoor map parameter
        self.obstacle_number = rospy.get_param("~obstacle_number")
        self.obstacle_extra_radius = rospy.get_param("~obstacle_extra_radius")

if __name__ == '__main__':
    rospy.init_node('map_generator')
    # if rospy.get_param("map_file") == "random_map":
    task_generator = MapGenerator()
    rospy.spin()