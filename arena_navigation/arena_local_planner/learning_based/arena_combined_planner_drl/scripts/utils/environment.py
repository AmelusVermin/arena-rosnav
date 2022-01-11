import gym
import rospy
import numpy as np
import time
import re
import nav_msgs

from flatland_msgs.srv import StepWorld, StepWorldRequest
from gym import spaces
from flatland_msgs.srv import StepWorld
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from task_generator.task_generator.tasks import *
from .geometry_utils import pose3D_to_pose2D, get_pose_difference, get_path_length
from .observer import Observer
from .reward import RewardCalculator

class FlatlandEnv(gym.Env):
    """ Custom environment that follows gym interface """

    def __init__(self, ns, args, paths, global_planner, mid_planner, train_mode=True, seed=1):
        t1 = time.perf_counter()
        super(FlatlandEnv, self).__init__()

        try:
            # given every environment enough time to initialize, if we dont put sleep,
            # the training script may crash.
            if(re.fullmatch("sim_([0-9]+)", ns) is not None):
                ns_int = int(ns.split("_")[1])-1
            else:
                # if the ns does not match the pattern, its probably the empty ns and thus, only one environment is created
                ns_int = 0
            time.sleep(ns_int * 7)
        except Exception:
            rospy.logwarn(
                f"Can't determine the number of the environment, training script may crash!"
            )

        # namespace creation
        self.ns = ns
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"
        self._steps_curr_episode = 0
        self._curr_episode = 0
        self._seed = seed
        self._train_mode = train_mode

        # init ros node for environment 
        if train_mode:
            rospy.init_node(f"train_env_{self.ns}", disable_signals=False, log_level=args.log_level)
        else:
            rospy.init_node(f"eval_env_{self.ns}", disable_signals=False, log_level=args.log_level)

        # set some arguments used outside the init method
        self._max_train_steps_per_episode = args.train_max_steps_per_episode
        self._max_eval_steps_per_episode = args.eval_max_steps_per_episode
        self._num_lidar_beams = args.num_lidar_beams
        self._extended_eval = args.extended_eval
        self._max_deque_size = args.max_deque_size
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
        self._global_planner = global_planner(self.ns)
        self._mid_planner = mid_planner(self.ns)
        self._replan = True
        # variables for storing global plans and mid plans
        assert args.global_planner_call_interval != 0 and args.mid_planner_call_interval != 0
        self._gp_interval = args.global_planner_call_interval
        self._mp_interval = args.mid_planner_call_interval
        self._last_global_plan = None
        self._last_subgoal = None

        #setup rewarding
        if args.safe_dist is None:
            safe_dist = 1.6 * args.robot_radius

        self.reward_calculator = RewardCalculator(
            robot_radius=args.robot_radius,
            safe_dist=safe_dist,
            goal_radius=args.goal_radius,
            rule=args.reward_fnc,
            collision_tolerance=args.collision_tolerance,
            extended_eval=self._extended_eval,
        )

        # create task 
        self._task_manager = get_predefined_task(self.ns, mode=args.task_mode, start_stage=args.task_curr_stage, PATHS=paths)

        # publisher for random map training
        self.demand_map_pub = rospy.Publisher("/demand", String, queue_size=1)

        # set up manual simulation stepping of flatland in training mode
        action_rate = rospy.get_param("/robot_action_rate")
        while not (type(action_rate) == float or type(action_rate) == int):
            # for some reason get_param returns sometimes not the right value and type TODO find reason for this
            rospy.logerr(f"rospy.get_param('/robot_action_rate') returns: {action_rate} in namespace: {self.ns}, try again!")
            action_rate = rospy.get_param("/robot_action_rate")

        self._action_frequency = 1 / action_rate
        self._is_sim_in_train_mode = rospy.get_param("train_mode")
        if self._is_sim_in_train_mode:
            self._service_name_step = f"{self.ns_prefix}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

        # let the environment run a bit to initiate some services in global and mid planner()
        if self._is_sim_in_train_mode:
            while not (self._global_planner.is_ready() and self._mid_planner.is_ready()): 
                self._call_service_takeSimStep(self._action_frequency)
        rospy.logdebug(f"{self.ns}: initialization finished!")
        print(time.perf_counter() - t1)

    def _pub_agent_action(self, action):
        """ publishes the agent action for the flatland sim """
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        self._agent_action_pub.publish(action_msg)

    def step(self, action):
        """ simulates an agent step in the environment and evaluates the action 
        done_reasons:  -1   -   not done
                        0   -   exceeded max steps
                        1   -   collision with obstacle
                        2   -   goal reached
        """
        rospy.logdebug(f"\n-------------ns:{self.ns}, step: {self._steps_curr_episode}, episode: {self._curr_episode}-------------")
        # publish agent action
        self._pub_agent_action(action)

        # make sure the flatland simulation steps forward
        if self._is_sim_in_train_mode:
            self._call_service_takeSimStep(self._action_frequency)
        else:
            try:
                rospy.wait_for_message(f"ns:{self.ns}, next_cycle", Bool)
            except Exception:
                pass
        

        # get current observation
        rospy.logdebug(f"ns:{self.ns}, get observation")
        obs_dict = self._observer.get_observation()
        scan = obs_dict['laser_scan']
        odom = obs_dict['odom']
        robot_pose_2D = obs_dict['robot_pose']
        global_goal = obs_dict['global_goal']

        # get global plan, check if valid and publish it
        if (self._steps_curr_episode % self._gp_interval) == 0:
            rospy.logdebug(f"replan global plan")
            global_plan = self._global_planner.get_global_plan(global_goal, odom)
            assert global_plan is not None, "global plan is None!"
            assert isinstance(global_plan, nav_msgs.msg.Path), "global path is not type of Path!"
            assert len(global_plan.poses) != 0, "global plan has length 0!"
            self._last_global_plan = global_plan
            self._global_planner_pub.publish(global_plan)
        assert self._last_global_plan is not None, "no global plan is available for this step!"

        # get subgoal, check if valid and publish it
        if (self._steps_curr_episode % self._mp_interval) == 0:
            rospy.logdebug(f"replan subgoal")
            subgoal = self._mid_planner.get_subgoal(global_plan, odom)
            assert subgoal is not None, "subgoal is None!"
            assert isinstance(subgoal, PoseStamped), "subgoal is not type of PoseStamped!"
            self._last_subgoal = subgoal
            self._mid_planner_pub.publish(subgoal)
        assert self._last_subgoal is not None, "no subgoal is available for this step!"

        # convert global plan Path message to nparray
        global_plan_array = Observer.process_global_plan_msg(self._last_global_plan)
        # prepare agent observation 
        global_plan_length =  get_path_length(global_plan_array)
        observation = self._observer.prepare_agent_observation(robot_pose_2D, scan, global_goal.pose, self._last_subgoal.pose, self._last_global_plan.poses, global_plan_length)
        #subgoal_2D = pose3D_to_pose2D(self._last_subgoal.pose)
        goal_tup = get_pose_difference(pose3D_to_pose2D(global_goal.pose), robot_pose_2D)
        subgoal_tup = get_pose_difference(pose3D_to_pose2D(self._last_subgoal.pose), robot_pose_2D)
        # calculate reward
        rospy.logdebug(f"ns:{self.ns}, get reward")
        reward, reward_info = self.reward_calculator.get_reward(
            observation[:self._num_lidar_beams],
            goal_tup,
            action=action,
            robot_pose=robot_pose_2D,
            subgoal_dist=subgoal_tup,
            global_plan=global_plan_array,
            global_plan_length=global_plan_length,
            episode_steps_passed=self._steps_curr_episode
        )

        rospy.logdebug(f"cum_reward: {reward}")

        # get done state
        done = reward_info["is_done"]

        # prepare additional infos
        info = {}
        info["done_reason"] = -1
        info["is_success"] = -1
        info["crash"] = reward_info["crash"]
        info["safe_dist"] = False
        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        # check if the max step number for an episode is reached
        if (self._train_mode and self._steps_curr_episode >= self._max_train_steps_per_episode or 
            not self._train_mode and self._steps_curr_episode >= self._max_eval_steps_per_episode):
                done = True
                info["done_reason"] = 0
                info["is_success"] = 0
        

        self._steps_curr_episode += 1
        rospy.logdebug(f"ns:{self.ns}, end of step, reward: {reward}, done: {done}, info: {info}")
        return observation, reward, done, info
    
    def reset(self):
        """ resets the environment for a new episode """
        rospy.loginfo(f"ns:{self.ns}, reset environment.")
        # demand a map update
        self.demand_map_pub.publish("") 
        # make the robot stop and stepforward
        self._agent_action_pub.publish(Twist())
        if self._is_sim_in_train_mode:
            self._sim_step_client()
        time.sleep(0.33)
        # reset deques to discard messages of old run and let the deque be filled in trainmode
        self._observer.reset()

        # reset task manager
        self._task_manager.reset()
        
        self.reward_calculator.reset()
        self._steps_curr_episode = 0
        self._curr_episode += 1
        
        obs_dict = None
        # as long as not all observations are available, let some time pass and try again
        # goal need to be available and at least one synced scan and odom in the new episode
        while(obs_dict is None):
            rospy.logdebug(f"ns:{self.ns}, waiting for observations.")
            if self._is_sim_in_train_mode:
                for _ in range(self._max_deque_size):
                    self._call_service_takeSimStep(self._action_frequency)
            else: 
                time.sleep(0.05)
            obs_dict = self._observer.get_observation()

        # extract observations
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
        # convert global plan Path message to nparray
        global_plan_array = Observer.process_global_plan_msg(global_plan)
        # prepare agent observation 
        observation = self._observer.prepare_agent_observation(robot_pose2D, scan, global_goal.pose, subgoal.pose, global_plan.poses, get_path_length(global_plan_array))
        rospy.logdebug(f"{self.ns}: reset environment finished!")
        return observation
    
    def render(self, mode='human'):
        pass

    def close (self):
        """ clean the environment when closing it """
        # close the planners
        self._global_planner.close()
        self._mid_planner.close()

    def _call_service_takeSimStep(self, t=None):
        """ let the flatland sim steps forward. Used in train mode """
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