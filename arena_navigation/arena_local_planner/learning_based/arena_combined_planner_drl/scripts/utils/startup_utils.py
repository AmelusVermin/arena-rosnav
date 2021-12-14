
import argparse
import os
import rospkg
import rosnode
import gym
import rospy
import time
import yaml
import json
import signal
import ctypes
from typing import Union
from stable_baselines.bench import Monitor
#from stable_baselines.common.utils import set_random_seed
from .environment import FlatlandEnv
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env.base_vec_env import VecEnv
from datetime import datetime as dt

PACKAGE_DIR = rospkg.RosPack().get_path("arena_combined_planner_drl")

def get_agent_name(args: argparse.Namespace) -> str:
    """Function to get agent name to save to/load from file system

    Example names:
    "MLP_B_64-64_P_32-32_V_32-32_relu_2021_01_07__10_32"
    "DRL_LOCAL_PLANNER_2021_01_08__7_14"

    :param args (argparse.Namespace): Object containing the program arguments
    """
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")
    agent_name = args.load_model
    if args.agent_type == "CUSTOM_MLP":
        agent_name = (
            "CUSTOM_MLP_S_"
            + args.shared_layers
            + "_P_"
            + args.policy_layer_sizes
            + "_V_"
            + args.value_layer_sizes
            + "_"
            + args.act_fn
            + "_"
            + START_TIME
        )
    if args.agent_type == "CUSTOM_CNN_LN_LSTM":
        agent_name = (
            "CUSTOM_CNN_LN_LSTM_S_"
            + args.shared_layers
            + "_P_"
            + args.policy_layer_sizes
            + "_V_"
            + args.value_layer_sizes
            + "_"
            + args.act_fn
            + "_"
            + START_TIME
        )
    if args.load_model:
        file_name = (args.model_path.split("/"))[-1]
        agent_name = file_name.split(".zip")[1]
    else:
        agent_name = args.agent_type + "_" + START_TIME
    return agent_name

def get_paths(agent_name: str, args: argparse.Namespace) -> dict:
    """
    Function to generate agent specific paths

    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("arena_local_planner_drl")

    PATHS = {
        "model": os.path.join(dir, "agents", agent_name),
        "tb": os.path.join(dir, "training_logs", "tensorboard", agent_name),
        "eval": os.path.join(dir, "training_logs", "train_eval_log", agent_name),
        "robot_setting": os.path.join(
            rospkg.RosPack().get_path("simulator_setup"),
            "robot",
            "myrobot" + ".model.yaml",
        ),
        "hyperparams": os.path.join(dir, "configs", "hyperparameters"),
        "robot_as": os.path.join(dir, "configs", "default_settings.yaml"),
        "curriculum": os.path.join(
            dir, "configs", "training_curriculum_map1small.yaml"
        ),
    }
    # check for mode
    if args.load is None:
        os.makedirs(PATHS["model"])
    elif not os.path.isfile(
        os.path.join(PATHS["model"], agent_name + ".zip")
    ) and not os.path.isfile(os.path.join(PATHS["model"], "best_model.zip")):
        raise FileNotFoundError(
            "Couldn't find model named %s.zip' or 'best_model.zip' in '%s'"
            % (agent_name, PATHS["model"])
        )
    # evaluation log enabled
    if args.eval_log:
        if not os.path.exists(PATHS["eval"]):
            os.makedirs(PATHS["eval"])
    else:
        PATHS["eval"] = None
    # tensorboard log enabled
    if args.tb:
        if not os.path.exists(PATHS["tb"]):
            os.makedirs(PATHS["tb"])
    else:
        PATHS["tb"] = None

    return PATHS

def unzip_map_parameters(paths: dict, numb_envs: int):
    if not os.path.exists(os.path.join(paths['map_folder'], 'tmp')):
        os.makedirs(os.path.join(paths['map_folder'], 'tmp'))
    with open(paths['map_parameters'], "r") as map_yaml:
        map_data = yaml.safe_load(map_yaml)
        for i in range(numb_envs):
            env_map_data = map_data[i+1]
            map_env_path = os.path.join(paths['map_folder'], 'tmp', "map_" + str(i) + ".json")
            with open(map_env_path, "w") as map_json:
                json.dump(env_map_data, map_json)


def setup_paths(args):
    """ setup the paths for saving logs and model """
    paths = {}
    paths['training'] = _setup_single_dir(args.train_log_dir, args.agent_name)
    paths['tensorboard'] = _setup_single_dir(args.tensorboard_log_dir, args.agent_name)
    paths['model'] = _setup_single_dir(args.model_save_dir, args.agent_name)
    paths['curriculum'] = _setup_single_dir(args.task_curriculum_path)
    #paths['map_folder'] = os.path.join(dir, 'configs', 'map_parameters')
    #paths['map_parameters'] = os.path.join(dir, 'configs', 'map_parameters', "map_curriculum_16envs.yaml")
    return paths

def _setup_single_dir(relative_path: str, agent_name:str=None):
    """ 
    Creates a full directory path based on the given realtive path and optionally the agent name 
    within the package and checks if the directory exists and creates it.

    relative_path (str): path to directory within the package
    agent_name (str): name/identifier of the agent (default None)
    """
    # check if a path is 
    if relative_path is not None:
        assert type(relative_path) is str
        if agent_name is not None:
            path = os.path.join(
                rospkg.RosPack().get_path("arena_combined_planner_drl"), 
                relative_path, 
                agent_name
            )
        else:
            path = os.path.join(
                rospkg.RosPack().get_path("arena_combined_planner_drl"), 
                relative_path
            )
        if not os.path.exists(path):
            os.makedirs(path)
    else: 
        path = None
    return path


def make_envs(
    args: argparse.Namespace,
    with_ns: bool,
    rank: int,
    global_planner,
    mid_planner,
    paths: dict,
    train: bool = True,
    
):
    """
    Utility function for multiprocessed env
    :param args: (Namespace) program arguments
    :param with_ns: (bool) if the system was initialized with namespaces
    :param rank: (int) index of the subprocess
    :param global_planner: (dynamically instantiated from config) global planner that shall be used during training 
    :param mid_planner: (dynamically instantiated from condig) mid planner that shall be used during training
    :param log_dir: (str) path to log directory for the agent
    :param seed: (int) the inital seed for RNG
    :param train: (bool) to differentiate between train and eval env
    
    :return: (Callable)
    """

    def _init() -> Union[gym.Env, gym.Wrapper]:
        train_ns = f"sim_{rank+1}" if with_ns else ""
        eval_ns = f"eval_sim" if with_ns else ""
        if train:
            # train env
            #paths['map_parameters'] = os.path.join(paths['map_folder'], 'tmp', "map_" + str(rank) + ".json")
            env = FlatlandEnv(train_ns, args, paths, global_planner, mid_planner)
        else:
            # eval env
            #paths['map_parameters'] = os.path.join(paths['map_folder'], "indoor_obs15.json")
            #seed = random.randint(1,1000)
            env = Monitor(
                FlatlandEnv(eval_ns, args, paths, global_planner, mid_planner),
                paths['training'],
                info_keywords=("done_reason", "is_success", "reached_subgoal", "crash", "safe_dist"),
            )
        return env
    return _init


def wait_for_nodes(with_ns: bool, n_envs: int, timeout: int = 30, nodes_per_ns: int = 3) -> None:
    """
    Checks for timeout seconds if all nodes to corresponding namespace are online.

    :param with_ns: (bool) if the system was initialized with namespaces
    :param n_envs: (int) number of virtual environments
    :param timeout: (int) seconds to wait for each ns
    :param nodes_per_ns: (int) usual number of nodes per ns
    """
    if with_ns:
        assert (with_ns and n_envs >= 1), f"Illegal number of environments parsed: {n_envs}"
    else:
        assert (not with_ns and n_envs == 1), f"Simulation setup isn't compatible with the given number of envs"

    for i in range(n_envs):
        for k in range(timeout):
            ns = "sim_" + str(i + 1) if with_ns else ""
            namespaces = rosnode.get_node_names(namespace=ns)

            if len(namespaces) >= nodes_per_ns:
                break

            rospy.logwarn(f"Check if all simulation parts of namespace '{ns}' are running properly")
            rospy.logwarn(f"Trying to connect again..")
            assert (k < timeout - 1), f"Timeout while trying to connect to nodes of '{ns}'"
            time.sleep(1)

def load_vec_normalize(args: argparse.Namespace, save_paths: dict, env: VecEnv, eval_env: VecEnv):
    if args.normalize:
        load_path = os.path.join(save_paths["model"], "vec_normalize.pkl")
        if os.path.isfile(load_path):
            env = VecNormalize.load(load_path=load_path, venv=env)
            eval_env = VecNormalize.load(load_path=load_path, venv=eval_env)
            print("Succesfully loaded VecNormalize object from pickle file..")
        else:
            env = VecNormalize(
                env, training=True, norm_obs=True, norm_reward=False, clip_reward=15
            )
            eval_env = VecNormalize(
                eval_env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
        return env, eval_env

def set_pdeathsig(sig = signal.SIGTERM):
    """ Used for sending signals to subprocess when parent process dies. (used as parameter in Popen) """
    libc = ctypes.CDLL("libc.so.6")
    def callable():
        return libc.prctl(1, sig)
    return callable