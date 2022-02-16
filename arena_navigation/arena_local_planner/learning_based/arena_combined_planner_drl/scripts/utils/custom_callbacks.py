import warnings
import rospy
import numpy as np
import time
import gym
import os

import tensorflow as tf
from typing import Union, Optional, Dict, Any
from std_msgs.msg import Bool
from stable_baselines.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv, VecNormalize
from stable_baselines.common.evaluation import evaluate_policy


class InitiateNewTrainStage(BaseCallback):
    """
    Introduces new training stage when threshhold reached.
    It must be used with "EvalCallback".

    :param treshhold_type (str): checks threshhold for either percentage of successful episodes (succ) or mean reward (rew)
    :param rew_threshold (int): mean reward threshold to trigger new stage
    :param succ_rate_threshold (float): threshold percentage of succesful episodes to trigger new stage
    :param task_mode (str): training task mode, if not 'staged' callback won't be called
    :param verbose:
    """

    def __init__(
        self,
        n_envs: int = 1,
        treshhold_type: str = "succ",
        upper_threshold: float = 0,
        lower_threshold: float = 0,
        task_mode: str = "staged",
        verbose=0,
    ):

        super(InitiateNewTrainStage, self).__init__(verbose=verbose)
        self.n_envs = n_envs
        self.threshhold_type = treshhold_type

        assert self.threshhold_type in {
            "rew",
            "succ",
        }, "given theshhold type neither 'rew' or 'succ'"

        # default values
        if self.threshhold_type == "rew" and upper_threshold == 0:
            self.upper_threshold = 13
            self.lower_threshold = 7
        elif self.threshhold_type == "succ" and upper_threshold == 0:
            self.upper_threshold = 0.85
            self.lower_threshold = 0.6
        else:
            self.upper_threshold = upper_threshold
            self.lower_threshold = lower_threshold

        assert (
            self.upper_threshold > self.lower_threshold
        ), "upper threshold has to be bigger than lower threshold"
        assert (
            self.upper_threshold >= 0 and self.lower_threshold >= 0
        ), "upper/lower threshold have to be positive numbers"
        if self.threshhold_type == "succ":
            assert (
                self.upper_threshold <= 1 and self.lower_threshold >= 0
            ), "succ thresholds have to be between [1.0, 0.0]"

        self.verbose = verbose
        self.activated = bool(task_mode == "staged")

        if self.activated:
            rospy.set_param("/last_stage_reached", False)
            self._instantiate_publishers()

            self._trigger = Bool()
            self._trigger.data = True

    def _instantiate_publishers(self):
        self._publishers_next = []
        self._publishers_previous = []

        self._publishers_next.append(
            rospy.Publisher(f"/eval_sim/next_stage", Bool, queue_size=1)
        )
        self._publishers_previous.append(
            rospy.Publisher(f"/eval_sim/previous_stage", Bool, queue_size=1)
        )

        for env_num in range(self.n_envs):
            self._publishers_next.append(
                rospy.Publisher(f"/sim_{env_num+1}/next_stage", Bool, queue_size=1)
            )
            self._publishers_previous.append(
                rospy.Publisher(f"/sim_{env_num+1}/previous_stage", Bool, queue_size=1)
            )

    def _on_step(self, EvalObject: EvalCallback) -> bool:
        assert isinstance(
            EvalObject, EvalCallback
        ), f"InitiateNewTrainStage must be called within EvalCallback"
        if self.activated:
            if EvalObject.n_eval_episodes < 20:
                warnings.warn(
                    "Only %d evaluation episodes considered for threshold monitoring,"
                    "results might not represent agent performance well"
                    % EvalObject.n_eval_episodes
                )
            self.log_curr_stage(EvalObject)
            if (
                self.threshhold_type == "rew"
                and EvalObject.best_mean_reward <= self.lower_threshold
            ) or (
                self.threshhold_type == "succ"
                and EvalObject.last_success_rate <= self.lower_threshold
            ):
                for i, pub in enumerate(self._publishers_previous):
                    pub.publish(self._trigger)
            if (
                self.threshhold_type == "rew"
                and EvalObject.best_mean_reward >= self.upper_threshold
            ) or (
                self.threshhold_type == "succ"
                and EvalObject.last_success_rate >= self.upper_threshold
            ):
                if not rospy.get_param("/last_stage_reached"):
                    EvalObject.best_mean_reward = -np.inf
                    EvalObject.last_success_rate = -np.inf

                for i, pub in enumerate(self._publishers_next):
                    pub.publish(self._trigger)
            
            

    def log_curr_stage(self, EvalObject: EvalCallback):
        time.sleep(1)
        curr_stage = rospy.get_param("/curr_stage", -1)
        EvalObject.tb_log_value("train_stage/stage_x=trainsteps", curr_stage, EvalObject.num_timesteps)
        EvalObject.tb_log_value("train_stage/stage_x=evalcalls", curr_stage, EvalObject.eval_calls)
        

class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param threshold_type: Which threshold type to consider for
        stoppage criteria (rew or succ)
    :param threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose:
    """

    def __init__(self, treshhold_type: str="rew", threshold: float=14.5, verbose: int = 0):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose=verbose)
        self.threshold_type = treshhold_type
        assert self.threshold_type == "rew" or self.threshold_type == "succ", "Threshold type must be 'rew' or 'succ'!"
        
        if self.threshold_type == "rew":
            assert threshold > 0, "Reward threshold must be positive"
        else:
            assert threshold >= 0.0 and threshold <= 1.0, "Success threshold must be within 0 to 1"
        self.threshold = threshold


    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EvalCallback``"
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        if rospy.get_param("/task_mode") != "staged":
            if self.threshold_type == "rew":
                continue_training = bool(self.parent.best_mean_reward < self.threshold)
            else:
                continue_training = bool(self.parent.last_success_rate < self.threshold)
        else:
            if self.threshold_type == "rew":
                continue_training = not bool(
                    self.parent.best_mean_reward >= self.threshold and
                    rospy.get_param("/last_stage_reached"))
            else:
                continue_training = not bool(
                    self.parent.last_success_rate >= self.threshold and
                    rospy.get_param("/last_stage_reached"))
        if self.verbose > 0 and not continue_training:
            if self.threshold_type == "rew":
                print(
                    f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                    f" is above the threshold {self.threshold}"
                )
            else:
                print(
                    f"Stopping training because the success rate {self.parent.last_success_rate:.2f} "
                    f" is above the threshold {self.threshold}"
                )
        return continue_training

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param train_env: The environment used for saving moving average of VecNormalize 
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        train_env: Union[gym.Env, VecEnv],
        callback_on_eval_end: Optional[BaseCallback] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.last_success_rate = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.callback_on_eval_end = callback_on_eval_end
        self.eval_calls = 0
        self.reached_subgoal = False
        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.train_env = train_env
        self.model_save_path = model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.model_save_path is not None:
            self.best_model_save_path = os.path.join(self.model_save_path, "best_model")
            self.newest_model_save_path = os.path.join(self.model_save_path, "newest_model")
        
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.newest_model_save_path is not None:
            os.makedirs(self.newest_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["_info"]
        # VecEnv: unpack
        if not isinstance(info, dict):
            info = info[0]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
        
            self._buffer_info_value(info, "time consumption", self._rw_time_consumption)
            self._buffer_info_value(info, "reduced path length", self._rw_path_length)
            self._buffer_info_value(info, "goal reached", self._rw_goal_reached)
            self._buffer_info_value(info, "goal approached", self._rw_goal_approached)
            self._buffer_info_value(info, "collision", self._rw_collision)
            self._buffer_info_value(info, "safe dist", self._rw_safe_dist)
            self._buffer_info_value(info, "not moving", self._rw_not_moving)
            self._buffer_info_value(info, "distance traveled", self._rw_distance_traveled)
            self._buffer_info_value(info, "distance global plan", self._rw_distance_global_plan)
            self._buffer_info_value(info, "following global plan", self._rw_following_global_plan)
            self._buffer_info_value(info, "abrupt direction change", self._rw_abrupt_direction_change)


    def _buffer_info_value(self, info: dict, key: str, buffer: list):
        val = info.get(key)
        if val is not None:
            buffer.append(val)
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            new_best = False
            self.eval_calls += 1
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []
            self._rw_time_consumption = []
            self._rw_path_length = []
            self._rw_goal_reached = []
            self._rw_goal_approached = []
            self._rw_collision = []
            self._rw_safe_dist = []
            self._rw_not_moving = []
            self._rw_distance_traveled = []
            self._rw_distance_global_plan = []
            self._rw_following_global_plan = []
            self._rw_abrupt_direction_change = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            mean_rw_time_consumption = np.mean(self._rw_time_consumption) if self._rw_time_consumption else 0
            mean_rw_path_length = np.mean(self._rw_path_length) if self._rw_path_length else 0
            mean_rw_goal_reached = np.mean(self._rw_goal_reached) if self._rw_goal_reached else 0
            mean_rw_goal_approached = np.mean(self._rw_goal_approached) if self._rw_goal_approached else 0
            mean_rw_collision = np.mean(self._rw_collision) if self._rw_collision else 0
            mean_rw_safe_dist = np.mean(self._rw_safe_dist) if self._rw_safe_dist else 0
            mean_rw_not_moving = np.mean(self._rw_not_moving) if self._rw_not_moving else 0
            mean_rw_distance_traveled = np.mean(self._rw_distance_traveled) if self._rw_distance_traveled else 0
            mean_rw_distance_global_plan = np.mean(self._rw_distance_global_plan) if self._rw_distance_global_plan else 0
            mean_rw_following_global_plan = np.mean(self._rw_following_global_plan) if self._rw_following_global_plan else 0
            mean_rw_abrupt_direction_change = np.mean(self._rw_abrupt_direction_change) if self._rw_abrupt_direction_change else 0
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.tb_log_value("eval_x=trainsteps/mean_reward", 
                float(mean_reward), self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps/mean_ep_length", 
                mean_ep_length, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/time_consumption", 
                mean_rw_time_consumption, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/reduced_path_length", 
                mean_rw_path_length, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/goal_reached", 
                mean_rw_goal_reached, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/goal_approached", 
                mean_rw_goal_approached, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/collision", 
                mean_rw_collision, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/safe_dist", 
                mean_rw_safe_dist, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/not_moving", 
                mean_rw_not_moving, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/distance_traveled", 
                mean_rw_distance_traveled, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/distance_global_plan", 
                mean_rw_distance_global_plan, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/following_global_plan", 
                mean_rw_following_global_plan, self.num_timesteps)
            self.tb_log_value("eval_x=trainsteps_rw_comp/abrupt_direction_change", 
                mean_rw_abrupt_direction_change, self.num_timesteps)

            self.tb_log_value("eval_x=evalcalls/mean_reward", 
                    float(mean_reward), self.eval_calls)
            self.tb_log_value("eval_x=evalcalls/mean_ep_length", 
                    mean_ep_length, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/time_consumption", 
                    mean_rw_time_consumption, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/path_length", 
                    mean_rw_path_length, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/goal_reached", 
                    mean_rw_goal_reached, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/goal_approached", 
                    mean_rw_goal_approached, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/collision", 
                    mean_rw_collision, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/safe_dist", 
                    mean_rw_safe_dist, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/not_moving", 
                    mean_rw_not_moving, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/distance_traveled", 
                    mean_rw_distance_traveled, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/distance_global_plan", 
                    mean_rw_distance_global_plan, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/following_global_plan", 
                    mean_rw_following_global_plan, self.eval_calls)
            self.tb_log_value("eval_x=evalcalls_rw_comp/abrupt_direction_change", 
                    mean_rw_abrupt_direction_change, self.eval_calls)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.tb_log_value("eval_x=trainsteps/success_rate", success_rate, self.num_timesteps)
                self.tb_log_value("eval_x=evalcalls/success_rate", success_rate, self.eval_calls)
                self.last_success_rate = success_rate

            # save best model
            print(f"new reward: {mean_reward}, best reward in this stage so far: {self.best_mean_reward}")
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    if isinstance(self.train_env, VecNormalize):
                        self.train_env.save(
                            os.path.join(self.best_model_save_path, "vec_normalize.pkl"))
                    with open(os.path.join(self.best_model_save_path, "save_best_model_timesteps.txt"), 'a+') as f:
                        f.write(f"n_calls: {self.n_calls}, eval_calls: {self.eval_calls}\n")
                self.best_mean_reward = mean_reward
                new_best = True
            
            # save newest model
            if self.model_save_path is not None:
                self.model.save(os.path.join(self.newest_model_save_path, "newest_model"))
                if isinstance(self.train_env, VecNormalize):
                    self.train_env.save(
                        os.path.join(self.newest_model_save_path, "vec_normalize.pkl"))
                with open(os.path.join(self.newest_model_save_path, "newest_best_model_timesteps.txt"), 'a+') as f:
                    f.write(f"n_calls: {self.n_calls}, eval_calls: {self.eval_calls}\n")

            if self.callback_on_eval_end is not None:
                self.callback_on_eval_end._on_step(self)

            if new_best:        
            # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

    def tb_log_value(self, tag: str , value, timestep):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.locals['writer'].add_summary(summary, timestep)