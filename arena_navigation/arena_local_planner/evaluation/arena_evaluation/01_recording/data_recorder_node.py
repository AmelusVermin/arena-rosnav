#!/usr/bin/env python3

# general packages
import time
import numpy as np
import csv
import os
import sys
import subprocess
import yaml

# ros packages
import rospy
from std_msgs.msg import Int16, Float32MultiArray, Float32
from geometry_msgs.msg import Pose2D, Pose, PoseWithCovarianceStamped, PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

# for transformations
from tf.transformations import euler_from_quaternion

class recorder():
    def __init__(self) -> None:
        # create rawdata csv file
        
        self.local_planner = rospy.get_param("local_planner", "test")
        self.start = time.time()
        self.dir_path = os.path.dirname(os.path.abspath(__file__)) # get path for current file, does not work if os.chdir() was used
        rospy.loginfo(f"ns:{self.dir_path}")
        self.model = rospy.get_param("model","base_model")
        self.now = time.strftime("%y-%m-%d_%H-%M-%S")
        #'''
        self.waypoint_generator = rospy.get_param("waypoint_generator", "test")
        self.record_only_planner = rospy.get_param("record_only_planner", False)
        self.scenario = rospy.get_param("scenario_file", "eval/test.json").replace(".json","").replace("eval/","")
        ''' #for debugging:
        self.waypoint_generator = True# rospy.get_param("waypoint_generator")
        self.record_only_planner = True#rospy.get_param("record_only_planner")
        self.scenario = 'test'# rospy.get_param("scenario_file").replace(".json","").replace("eval/","")
        #self.real-eval = rospy.get_param("real-eval")
        #'''
        
        if self.record_only_planner:
            with open(self.dir_path+"/{0}_{1}--{2}--{3}.csv".format(self.local_planner,self.model,self.scenario,self.now), "w+", newline = "") as file:
                writer = csv.writer(file, delimiter = ',')
                header = [["episode","time", "ep_time","laser_scan","robot_lin_vel_x","robot_lin_vel_y","robot_ang_vel","robot_orientation","robot_pos_x","robot_pos_y","action", "goal_pos_x", "goal_pos_y", "num_collisions", "reached_goal", "rew_fgp", "rew_dgp", "resp_time_full", "resp_time_min", "obs_x", "obs_y", "model"]]
                writer.writerows(header)
                file.close()
        else:
            with open(self.dir_path+"/{0}_{1}_{2}--{3}--{4}.csv".format(self.local_planner,self.waypoint_generator,self.model,self.scenario,self.now), "w+", newline = "") as file:
                writer = csv.writer(file, delimiter = ',')
                header = [["episode","time", "ep_time","laser_scan","robot_lin_vel_x","robot_lin_vel_y","robot_ang_vel","robot_orientation","robot_pos_x","robot_pos_y","action", "goal_pos_x", "goal_pos_y", "num_collisions", "reached_goal", "rew_fgp", "rew_dgp", "resp_time", "resp_time_min", "obs_x", "obs_y", "model"]]
                writer.writerows(header)
                file.close()

        # read config
        with open(self.dir_path+ "/data_recorder_config.yaml") as file:
            self.config = yaml.safe_load(file)

        # initialize variables to be recorded with default values, NOTE: time is recorded as well, but no need for seperate variable
        self.episode = 0
        self.laserscan = ["None"]
        self.robot_lin_vel_x = 0
        self.robot_lin_vel_y = 0
        self.robot_ang_vel = 0
        self.robot_orientation = 0
        self.robot_pos_x = 0
        self.robot_pos_y = 0
        self.action = ["None"]
        self.last_action_time = rospy.get_time()
        self.ep_start_time = rospy.get_time()
        self.curr_goal_x = None
        self.curr_goal_y = None
        self.num_collisions = 0
        self.reached_goal = False
        self.added_goal_state = False
        self.follow_global_plan_rew = 0.0
        self.dist_global_plam_rew = 0.0
        self.response_times_full = []
        self.response_times_min = []
        # subscribe to topics
        rospy.Subscriber("/scenario_reset", Int16, self.episode_callback)
        rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)
        rospy.Subscriber("/odom", Odometry, self.odometry_callback)
        rospy.Subscriber("/cmd_vel", Twist, self.action_callback)
        rospy.Subscriber("/goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/reward", Float32MultiArray, self.reward_callback)
        rospy.Subscriber("/response_time", Float32MultiArray, self.response_time_callback)

        self.obstacle_positions_x = {}
        self.obstacle_positions_y = {}

        self.obstacle_sub_00 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_00", MarkerArray, self.obstacle_callback_00)
        self.obstacle_sub_01 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_01", MarkerArray, self.obstacle_callback_01)
        self.obstacle_sub_02 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_02", MarkerArray, self.obstacle_callback_02)
        self.obstacle_sub_03 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_03", MarkerArray, self.obstacle_callback_03)
        self.obstacle_sub_04 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_04", MarkerArray, self.obstacle_callback_04)
        self.obstacle_sub_05 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_05", MarkerArray, self.obstacle_callback_05)
        self.obstacle_sub_06 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_06", MarkerArray, self.obstacle_callback_06)
        self.obstacle_sub_07 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_07", MarkerArray, self.obstacle_callback_07)
        self.obstacle_sub_08 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_08", MarkerArray, self.obstacle_callback_08)
        self.obstacle_sub_09 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_09", MarkerArray, self.obstacle_callback_09)
        self.obstacle_sub_10 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_10", MarkerArray, self.obstacle_callback_10)
        self.obstacle_sub_11 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_11", MarkerArray, self.obstacle_callback_11)
        self.obstacle_sub_12 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_12", MarkerArray, self.obstacle_callback_12)
        self.obstacle_sub_13 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_13", MarkerArray, self.obstacle_callback_13)
        self.obstacle_sub_14 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_14", MarkerArray, self.obstacle_callback_14)
        self.obstacle_sub_15 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_15", MarkerArray, self.obstacle_callback_15)
        self.obstacle_sub_16 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_16", MarkerArray, self.obstacle_callback_16)
        self.obstacle_sub_17 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_17", MarkerArray, self.obstacle_callback_17)
        self.obstacle_sub_18 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_18", MarkerArray, self.obstacle_callback_18)
        self.obstacle_sub_19 = rospy.Subscriber("/flatland_server/debug/model/obstacle_random_dynamic_19", MarkerArray, self.obstacle_callback_19)



        #rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

    def obstacle_callback_00(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["00"] = x
        self.obstacle_positions_y["00"] = y

    def obstacle_callback_01(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["01"] = x
        self.obstacle_positions_y["01"] = y

    def obstacle_callback_02(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["02"] = x
        self.obstacle_positions_y["02"] = y

    def obstacle_callback_03(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["03"] = x
        self.obstacle_positions_y["03"] = y

    def obstacle_callback_04(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["04"] = x
        self.obstacle_positions_y["04"] = y

    def obstacle_callback_05(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["05"] = x
        self.obstacle_positions_y["05"] = y

    def obstacle_callback_06(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["06"] = x
        self.obstacle_positions_y["06"] = y
    
    def obstacle_callback_07(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["07"] = x
        self.obstacle_positions_y["07"] = y

    def obstacle_callback_08(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["08"] = x
        self.obstacle_positions_y["08"] = y

    def obstacle_callback_09(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["09"] = x
        self.obstacle_positions_y["09"] = y

    def obstacle_callback_10(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["10"] = x
        self.obstacle_positions_y["10"] = y

    def obstacle_callback_11(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["1"] = x
        self.obstacle_positions_y["11"] = y

    def obstacle_callback_12(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["12"] = x
        self.obstacle_positions_y["12"] = y

    def obstacle_callback_13(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["13"] = x
        self.obstacle_positions_y["13"] = y

    def obstacle_callback_14(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["14"] = x
        self.obstacle_positions_y["14"] = y

    def obstacle_callback_15(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["15"] = x
        self.obstacle_positions_y["15"] = y

    def obstacle_callback_16(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["16"] = x
        self.obstacle_positions_y["16"] = y

    def obstacle_callback_17(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["17"] = x
        self.obstacle_positions_y["17"] = y

    def obstacle_callback_18(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["18"] = x
        self.obstacle_positions_y["18"] = y

    def obstacle_callback_19(self, msg):
        x = msg.markers[0].pose.position.x
        y = msg.markers[0].pose.position.y

        self.obstacle_positions_x["19"] = x
        self.obstacle_positions_y["19"] = y


    def clear_costmaps(self):
        bashCommand = "rosservice call /move_base/clear_costmaps"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error

    def response_time_callback(self, msg):
        self.response_times_full.append(msg.data[0])
        self.response_times_min.append(msg.data[1])

    def reward_callback(self, msg):
        self.follow_global_plan_rew += msg.data[0]
        self.dist_global_plam_rew += msg.data[1]

    def goal_callback(self, msg_pose):
        self.curr_goal_x = float("{:.4f}".format(msg_pose.pose.position.x))
        self.curr_goal_y = float("{:.4f}".format(msg_pose.pose.position.y))

    # define callback function for all variables and their respective topics
    def episode_callback(self, msg_scenario_reset: Int16):
        self.ep_start_time = rospy.get_time()
        self.episode = msg_scenario_reset.data
        #self.clear_costmaps()
        self.num_collisions = 0
        self.reached_goal = False
        self.added_goal_state = False
        self.follow_global_plan_rew = 0.0
        self.dist_global_plam_rew = 0.0
        self.response_times_full = []
        self.response_times_min = []

    def laserscan_callback(self, msg_laserscan: LaserScan):
        self.laserscan = [float("{:.4f}".format(min(msg_laserscan.ranges)))]
        # if min(msg_laserscan.ranges) <= 0.3:
        #     self.num_collisions += 1


        #  check for termination criterion "max time"
        # if time.time()-self.start > self.config["max_time"]:
        #     subprocess.call(["killall","-9","rosmaster"]) # apt-get install psmisc necessary
        #     sys.exit()

    def odometry_callback(self, msg_Odometry: Odometry):
        pose3d = msg_Odometry.pose.pose
        twist = msg_Odometry.twist.twist
        pose2d = self.pose3D_to_pose2D(pose3d)
        self.robot_lin_vel_x = float("{:.4f}".format(twist.linear.x))
        self.robot_lin_vel_y = float("{:.4f}".format(twist.linear.y))
        self.robot_ang_vel = float("{:.4f}".format(twist.angular.z))
        #if rospy.get_param("/real-eval", default=True):
        self.robot_orientation = float("{:.4f}".format(pose2d.theta))
        self.robot_pos_x = float("{:.4f}".format(pose2d.x))
        self.robot_pos_y =  float("{:.4f}".format(pose2d.y))

        if self.curr_goal_x is not None and self.curr_goal_y is not None:
            err_g = (self.robot_pos_x-self.curr_goal_x)**2+(self.robot_pos_y-self.curr_goal_y)**2
            if err_g < 1.0:
                self.reached_goal = True

    # def amcl_callback(self, msg_PoseWithCovarianceStamped: PoseWithCovarianceStamped):
    #     pose3d = msg_PoseWithCovarianceStamped.pose.pose
    #     pose2d = self.pose3D_to_pose2D(pose3d)
    #     #if rospy.get_param("/real-eval", default=False):
    #         self.robot_orientation = pose2d.theta
    #         self.robot_pos_x = pose2d.x
    #         self.robot_pos_y = pose2d.y

    def pose3D_to_pose2D(self, pose3d: Pose):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

    def action_callback(self, msg_action: Twist): # variables will be written to csv whenever an action is published
        self.action = [msg_action.linear.x,msg_action.linear.y,msg_action.angular.z]
        self.action = [float("{:.4f}".format(_)) for _ in self.action]
        current_simulation_action_time = rospy.get_time() 
        current_action_time = time.time()
        if current_simulation_action_time - self.last_action_time >= self.config["record_frequency"]:
            if self.laserscan != ["None"]:                
                self.last_action_time = current_simulation_action_time
                if min(self.laserscan) <= 0.3:
                    self.num_collisions += 1
                if True:
                    self.addData(np.array(
                        [self.episode,
                        float("{:.4f}".format(current_simulation_action_time)),
                        float("{:.4f}".format(current_simulation_action_time - self.ep_start_time)),
                        list(self.laserscan),
                        self.robot_lin_vel_x,
                        self.robot_lin_vel_y,
                        self.robot_ang_vel,
                        self.robot_orientation,
                        self.robot_pos_x,
                        self.robot_pos_y,
                        self.action,
                        self.curr_goal_x,
                        self.curr_goal_y,
                        self.num_collisions,
                        self.reached_goal,
                        float("{:.4f}".format(self.follow_global_plan_rew)),
                        float("{:.4f}".format(self.dist_global_plam_rew)),
                        float("{:.6f}".format(np.mean(self.response_times_full))),
                        float("{:.6f}".format(np.mean(self.response_times_min))),
                        list(self.obstacle_positions_x.values()),
                        list(self.obstacle_positions_y.values()),
                        self.model]
                    ))
                    if self.reached_goal and not self.added_goal_state:
                        self.added_goal_state = True

        # check for termination criterion "max episodes"
        if self.episode == self.config["max_episodes"]:
            subprocess.call(["killall","-9","rosmaster"]) # apt-get install psmisc necessary
            sys.exit()

    def addData(self, data:np.array): #add new row to the csv file
        if self.record_only_planner:
            with open(self.dir_path+"/{0}_{1}--{2}--{3}.csv".format(self.local_planner,self.model,self.scenario,self.now), "a+", newline = "") as file:
                writer = csv.writer(file, delimiter = ',') # writer has to be defined again for the code to work
                writer.writerows(data.reshape(1,-1)) # reshape into line vector
                file.close()
        else:
            with open(self.dir_path+"/{0}_{1}_{2}--{3}--{4}.csv".format(self.local_planner,self.waypoint_generator,self.model,self.scenario,self.now), "a+", newline = "") as file:
                writer = csv.writer(file, delimiter = ',') # writer has to be defined again for the code to work
                writer.writerows(data.reshape(1,-1)) # reshape into line vector
                file.close()



if __name__=="__main__":
    rospy.init_node("data_recorder")    
    data_recorder = recorder()
    rospy.spin()