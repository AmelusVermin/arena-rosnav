import rospy
import os
import pickle
from stable_baselines.ppo2 import PPO2
from geometry_msgs.msg import Twist, PoseStamped
from .utils.observer import Observer


class AgentNode():
    def __init__(self, ns, args):
        self.ns_prefix = "" if (ns == "" or ns is None) else f"/{ns}/"
        self._observer = Observer(ns, args)   
        self._publish_rate = rospy.Rate(rospy.get_param("publish_rate", 20))
        self._action_publisher = rospy.Publisher(
            f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)
        self._model, self._n_env = self._load_model()
        self._normalize = self._load_pkl()

    def _load_pkl(self, vec_norm_file):
        """ loads the normalize function from pickle file if given """
        normalize_func = None
        if vec_norm_file:
            assert os.path.isfile(
                vec_norm_file
            ), f"VecNormalize file cannot be found at {vec_norm_file}!"

            with open(vec_norm_file, "rb") as file_handler:
                vec_normalize = pickle.load(file_handler)
            normalize_func = vec_normalize.normalize_obs
        return normalize_func

    def _load_model(self, model_zip):
        """ Loads the trained policy """
        
        assert os.path.isfile(
            model_zip
        ), f"Compressed model cannot be found at {model_zip}!"
        ppo_model = PPO2.load(model_zip)
        return ppo_model.policy, ppo_model.n_env
        
    def _prepare_action_msg(self, action):
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        return action_msg

    def run_node(self):
        states = None
        done_masks = [False for _ in range(self._n_env)]
        while not rospy.is_shutdown():
            # get observations
            obs_dict = self._observer.get_deployment_observation()
            proc_obs = self._observer.get_processed_observation(obs_dict)
            # normalize observation of normalize function is given
            if self._normalize:
                proc_obs = self._normalize(proc_obs)
            # get action 
            actions, states = self._model.predict(proc_obs, states, done_masks, deterministic=True)
            #publish action
            action_msg = self._prepare_action_msg(actions[0])
            self._action_publisher.publish(action_msg)
            # sleep in given rate
            self._publish_rate.sleep()

def parse_args():
    args = None
    return args

if __name__ == "__main__":
    args = parse_args()
    ns = rospy.get_param("ns", "")
    agent_node = AgentNode(ns, args)
    agent_node.run_node()