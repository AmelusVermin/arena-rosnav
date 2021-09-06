import gym
import rospy
import numpy as np
import sys
import time
from flatland_msgs.srv import StepWorld, StepWorldRequest
from gym import spaces
from flatland_msgs.srv import StepWorld
from geometry_msgs.msg import Pose2D, Twist, PoseStamped
import nav_msgs
from std_msgs.msg import String
from task_generator.task_generator.tasks import *
from utils.observer import Observer

class FlatlandEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, ns, args, global_planner, mid_planner):
        super(FlatlandEnv, self).__init__()
        # namespace creation
        self.ns = ns
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"
        self._step = 0
        self._num_lidar_beams = args.num_lidar_beams
        # observer related inits
        self._observer = Observer(ns, args)
        self.observation_space = self._observer.get_agent_observation_space()
        # action space of agent (local planner)
        self.action_space = spaces.Box(
            low=np.array([args.linear_range[0], args.angular_range[0]]),
            high=np.array([args.linear_range[1], args.angular_range[1]]),
            dtype=np.float,
        )
        # define publisher
        self._agent_action_pub = rospy.Publisher(f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)
        self._global_planner_pub = rospy.Publisher(f"{self.ns_prefix}globalPlan", nav_msgs.msg.Path, queue_size=1)
        self._mid_planner_pub = rospy.Publisher(f"{self.ns_prefix}subgoal", PoseStamped, queue_size=1)

        # set global and mid planner
        self._global_planner = global_planner
        self._mid_planner = mid_planner
        
        # create task
        self._task = get_predefined_task(ns, mode="random")

        # set up manual simulation stepping of flatland in training mode
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")
        self._is_train_mode = rospy.get_param("/train_mode")
        if self._is_train_mode:
            self._service_name_step = f"{self.ns_prefix}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

        # publisher for random map training
        self.demand_map_pub = rospy.Publisher("/demand", String, queue_size=1)

    def _pub_agent_action(self, action):
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        self._agent_action_pub.publish(action_msg)

    def _pub_global_path(self):
        pass

    def _pub_subgoal(self):
        pass

    def step(self, action):
        rospy.logdebug(f"\n-------------------------------------------------------\nstep: {self._step}")
        # publish agent action
        self._pub_agent_action(action)

        # make sure the flatland simulation steps forward
        if self._is_train_mode:
            self._call_service_takeSimStep(self._action_frequency)
        else:
            try:
                rospy.wait_for_message(f"{self.ns_prefix}next_cycle", Bool)
            except Exception:
                pass

        # get current observation
        obs_dict = self._observer.get_observation()
        scan = obs_dict['laser_scan']
        odom = obs_dict['odom']
        robot_pose2D = obs_dict['robot_pose']
        global_goal = obs_dict['global_goal']
                
        # get global plan and publish it
        global_plan = self._global_planner.get_global_plan(global_goal, odom)
        self._global_planner_pub.publish(global_plan)
        # get subgoal and publish it
        subgoal = self._mid_planner.get_subgoal(global_plan, odom)
        self._mid_planner_pub.publish(subgoal)

        # prepare agent observation 
        observation = self._observer.prepare_agent_observation(scan, subgoal.pose, robot_pose2D)

        # TODO calculate reward
        reward = 0
        # TODO get done state
        done = 0
        # TODO prepare additional infos
        info = None
        self._step = self._step + 1
        return observation, reward, done, info
    
    def reset(self):
        # demand a map update
        self.demand_map_pub.publish("") 
        # make the robot stop and stepforward
        self._agent_action_pub.publish(Twist())
        if self._is_train_mode:
            self._sim_step_client()
        time.sleep(0.33)
        #reset task to generate a new goal and starting pos
        self._task.reset()
        # get first observations
        obs_dict = self._observer.get_observation()
        scan = obs_dict['laser_scan']
        odom = obs_dict['odom']
        robot_pose2D = obs_dict['robot_pose']
        global_goal = obs_dict['global_goal']
                
        # get global plan and publish it
        global_plan = self._global_planner.get_global_plan(global_goal, odom)
        self._global_planner_pub.publish(global_plan)
        # get subgoal and publish it
        subgoal = self._mid_planner.get_subgoal(global_plan, odom)
        self._mid_planner_pub.publish(subgoal)

        # prepare agent observation 
        observation = self._observer.prepare_agent_observation(scan, subgoal.pose, robot_pose2D)
        return observation
    
    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def _call_service_takeSimStep(self, t=None):
        request = StepWorldRequest() if t is None else StepWorldRequest(t)
        timeout = 12
        try:
            for i in range(timeout):
                response = self._sim_step_client(request)
                rospy.logdebug(f"step service={response}")

                if response.success:
                    break
                if i == timeout - 1:
                    raise TimeoutError(
                        f"Timeout while trying to call '{self.ns_prefix}step_world'"
                    )
                time.sleep(0.33)

        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)