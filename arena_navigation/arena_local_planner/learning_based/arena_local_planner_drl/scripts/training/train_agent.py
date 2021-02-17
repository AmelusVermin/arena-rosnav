import os, sys
# sys.path.insert(0, os.path.abspath(".."))
# for me the python path must be added manuelly , could be deleted directly if your machine could identify the corresponding paths
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/arena_navigation')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/arena_navigation/arena_local_planner')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/arena_navigation/arena_local_planner/learning_based/')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/arena_navigation/arena_local_planner/learning_based/arena_local_planner_drl/scripts')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/task_generator')
sys.path.append('/home/junhui/study/Masterarbeit/arenarosnav/test_ws/src/arena_rosnav/task_generator/task_generator')
import rospy
import time
import rosnode
from typing import Union
from datetime import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed

from task_generator.task_generator.tasks import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.scripts.custom_policy import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.rl_agent.envs.flatland_gym_env import FlatlandEnv
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.argsparser import parse_training_args
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.train_agent_utils import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.custom_mlp_utils import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.staged_train_callback import InitiateNewTrainStage

##### HYPERPARAMETER #####
""" will be used upon initializing new agent """
robot = "myrobot"
gamma = 0.99
n_steps = 4000
ent_coef = 0.005
learning_rate = 3e-4
vf_coef = 0.2
max_grad_norm = 0.5
gae_lambda = 0.95
batch_size = 16
n_epochs = 3
clip_range = 0.2
reward_fnc = "rule_01"
discrete_action_space = False
normalize = True
start_stage = 1
train_max_steps_per_episode = 500
eval_max_steps_per_episode = 500
goal_radius = 0.25
task_mode = "random"    # custom, random or staged
normalize = True
##########################


def get_agent_name(args) -> str:
    """ Function to get agent name to save to/load from file system
    
    Example names:
    "MLP_B_64-64_P_32-32_V_32-32_relu_2021_01_07__10_32"
    "DRL_LOCAL_PLANNER_2021_01_08__7_14"

    :param args (argparse.Namespace): Object containing the program arguments
    """
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

    if args.custom_mlp:
        return (
            "MLP_B_" + args.body 
            + "_P_" + args.pi 
            + "_V_" + args.vf + "_" 
            + args.act_fn + "_" + START_TIME)
    if args.load is None:
        return args.agent + "_" + START_TIME
    return args.load


def get_paths(agent_name: str, args) -> dict:
    """ Function to generate agent specific paths 
    
    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path('arena_local_planner_drl')

    PATHS = {
        'model': 
            os.path.join(
                dir, 'agents', agent_name),
        'tb': 
            os.path.join(
                dir, 'training_logs', 'tensorboard', agent_name),
        'eval': 
            os.path.join(
                dir, 'training_logs', 'train_eval_log', agent_name),
        'robot_setting': 
            os.path.join(
                rospkg.RosPack().get_path('simulator_setup'),
                'robot', robot + '.model.yaml'),
        'robot_as': 
            os.path.join(
                dir, 'configs', 'default_settings.yaml'),
        'curriculum': 
            os.path.join(
                dir, 'configs', 'training_curriculum.yaml')
    }
    # check for mode
    if args.load is None:
        os.makedirs(PATHS.get('model'))
    else:
        if (not os.path.isfile(
                os.path.join(PATHS.get('model'), AGENT_NAME + ".zip")) 
            and not os.path.isfile(
                os.path.join(PATHS.get('model'), "best_model.zip"))
            ):
            raise FileNotFoundError(
                "Couldn't find model named %s.zip' or 'best_model.zip' in '%s'" 
                % (AGENT_NAME, PATHS.get('model')))
    # evaluation log enabled
    if args.eval_log:
        if not os.path.exists(PATHS.get('eval')):
            os.makedirs(PATHS.get('eval'))
    else:
        PATHS['eval'] = None
    # tensorboard log enabled
    if args.tb:
        if not os.path.exists(PATHS.get('tb')):
            os.makedirs(PATHS.get('tb'))
    else:
        PATHS['tb'] = None

    return PATHS


def make_envs(task_manager: Union[RandomTask, StagedRandomTask, ManualTask, ScenerioTask], 
              rank: int, 
              params: dict, 
              seed: int=0, 
              PATHS: dict=None, 
              train: bool=True):
    """
    Utility function for multiprocessed env
    
    :param task_manager: (Object) interface for managing the tasks
    :param rank: (int) index of the subprocess
    :param params: (dict) hyperparameters of agent to be trained
    :param seed: (int) the inital seed for RNG
    :param PATHS: (dict) script relevant paths
    :param train: (bool) to differentiate between train and eval env
    :param args: (Namespace) program arguments
    :return: (Callable)
    """
    def _init() -> Union[gym.Env, gym.Wrapper]:
        if train:
            # train env
            # if rank+1 < 10:
            #     suffix = '0'+str(rank+1)
            # else:
            #     suffix = str(rank+1)
            env = FlatlandEnv(
                f"sim_{rank+1}", task_manager, 
                PATHS.get('robot_setting'), PATHS.get('robot_as'), 
                params['reward_fnc'], params['discrete_action_space'], 
                goal_radius=params['goal_radius'], 
                max_steps_per_episode=params['train_max_steps_per_episode'],
                debug=args.debug)
        else:
            # eval env
            env = Monitor(
                FlatlandEnv(
                    f"sim_{rank+1}", task_manager, 
                    PATHS.get('robot_setting'), PATHS.get('robot_as'), 
                    params['reward_fnc'], params['discrete_action_space'], 
                    goal_radius=params['goal_radius'], 
                    max_steps_per_episode=params['eval_max_steps_per_episode'], 
                    train_mode=False, debug=args.debug
                    ),
                PATHS.get('eval'), info_keywords=("done_reason", "is_success"))
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    args, _ = parse_training_args()

    if args.debug:
        rospy.init_node("debug_node", disable_signals=False)
        
    # generate agent name and model specific paths
    AGENT_NAME = get_agent_name(args)
    PATHS = get_paths(AGENT_NAME, args)

    print("________ STARTING TRAINING WITH:  %s ________\n" % AGENT_NAME)
    # check if simulations are booted
    for i in range(args.n_envs):
        ns = rosnode.get_node_names(namespace='sim_'+str(i+1))
        assert (len(ns) > 0
        ), f"Check if {args.n_envs} different simulation environments are running"
        assert (len(ns) > 2
        ), f"Check if all simulation parts of namespace '{'/sim_'+str(i+1)}' are running properly"

    # initialize hyperparameters (save to/ load from json)
    hyperparams_obj = agent_hyperparams(
        AGENT_NAME, robot, gamma, n_steps, ent_coef, 
        learning_rate, vf_coef,max_grad_norm, gae_lambda, batch_size, 
        n_epochs, clip_range, reward_fnc, discrete_action_space, normalize, 
        task_mode, start_stage, train_max_steps_per_episode,
        eval_max_steps_per_episode, goal_radius)

    params = initialize_hyperparameters(
        agent_name=AGENT_NAME,           PATHS=PATHS, 
        hyperparams_obj=hyperparams_obj, load_target=args.load)

    # task managers for each simulation
    task_managers=[]
    for i in range(args.n_envs):
        # print("n",args.n_envs)
        # suffix=''
        # if i+1 < 10:
        #     suffix = '0'+str(i+1)
        # else:
        #     suffix = str(i+1)
        task_managers.append(
            get_predefined_task(
                f"sim_{i+1}", params['task_mode'], params['curr_stage'], PATHS))

    # instantiate gym environment
    # when debug run on one process only
    if not args.debug:
        env = SubprocVecEnv(
            [make_envs(task_managers[i], i, params=params, PATHS=PATHS) 
                for i in range(args.n_envs)], 
            start_method='fork')
    else:
        env = DummyVecEnv(
            [make_envs(task_managers[i], i, params=params, PATHS=PATHS) 
                for i in range(args.n_envs)])

    if params['normalize']:
        env = VecNormalize(
            env, training=True, 
            norm_obs=True, norm_reward=False, clip_reward=15)

    # threshold settings for training curriculum
    # type can be either 'succ' or 'rew'
    trainstage_cb = InitiateNewTrainStage(
        TaskManagers=task_managers, 
        treshhold_type="succ", 
        upper_threshold=0.9, lower_threshold=0.6, 
        task_mode=params['task_mode'], verbose=1)
    
    # stop training on reward threshold callback
    stoptraining_cb = StopTrainingOnRewardThreshold(
        reward_threshold=6, task_manager=task_managers[0], verbose=1)

    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    eval_env = DummyVecEnv(
        [make_envs(task_managers[0], 0, params=params, PATHS=PATHS, train=False)])

    if params['normalize']:
        eval_env = VecNormalize(
            eval_env, training=False, 
            norm_obs=True, norm_reward=False, clip_reward=15)
    
    # evaluation settings
    # n_eval_episodes: number of episodes to evaluate agent on
    # eval_freq: evaluate the agent every eval_freq train timesteps
    eval_cb = EvalCallback(
        eval_env, 
        n_eval_episodes=35,         eval_freq=25000, 
        log_path=PATHS.get('eval'), best_model_save_path=PATHS.get('model'), 
        deterministic=True,         callback_on_eval_end=trainstage_cb,
        callback_on_new_best=stoptraining_cb)
   
    # determine mode
    if args.custom_mlp:
        # custom mlp flag
        model = PPO(
            "MlpPolicy", env, 
            policy_kwargs = dict(
                net_arch = args.net_arch, activation_fn = get_act_fn(args.act_fn)), 
            gamma = gamma,                     n_steps = n_steps, 
            ent_coef = ent_coef,               learning_rate = learning_rate, 
            vf_coef = vf_coef,                 max_grad_norm = max_grad_norm, 
            gae_lambda = gae_lambda,           batch_size = batch_size, 
            n_epochs = n_epochs,               clip_range = clip_range, 
            tensorboard_log = PATHS.get('tb'), verbose = 1
        )
    elif args.agent is not None:
        # predefined agent flag
        if args.agent == "MLP_ARENA2D":
                model = PPO(
                    MLP_ARENA2D_POLICY, env, 
                    gamma = gamma,                     n_steps = n_steps, 
                    ent_coef = ent_coef,               learning_rate = learning_rate, 
                    vf_coef = vf_coef,                 max_grad_norm = max_grad_norm, 
                    gae_lambda = gae_lambda,           batch_size = batch_size, 
                    n_epochs = n_epochs,               clip_range = clip_range, 
                    tensorboard_log = PATHS.get('tb'), verbose = 1
                )

        elif args.agent == "DRL_LOCAL_PLANNER" or args.agent == "CNN_NAVREP":
            if args.agent == "DRL_LOCAL_PLANNER":
                policy_kwargs = policy_kwargs_drl_local_planner
            else:
                policy_kwargs = policy_kwargs_navrep

            model = PPO(
                "CnnPolicy", env, 
                policy_kwargs = policy_kwargs, 
                gamma = gamma,                     n_steps = n_steps, 
                ent_coef = ent_coef,               learning_rate = learning_rate, 
                vf_coef = vf_coef,                 max_grad_norm = max_grad_norm, 
                gae_lambda = gae_lambda,           batch_size = batch_size, 
                n_epochs = n_epochs,               clip_range = clip_range, 
                tensorboard_log = PATHS.get('tb'), verbose = 1
            )
    else:
        # load flag
        if os.path.isfile(
                os.path.join(PATHS.get('model'), AGENT_NAME + ".zip")):
            model = PPO.load(
                os.path.join(PATHS.get('model'), AGENT_NAME), env)
        elif os.path.isfile(
                os.path.join(PATHS.get('model'), "best_model.zip")):
            model = PPO.load(
                os.path.join(PATHS.get('model'), "best_model"), env)
        update_hyperparam_model(model, params, args.n_envs)

    # set num of timesteps to be generated
    if args.n is None:
<<<<<<< HEAD
        n_timesteps = 40000000
=======
        n_timesteps = 20000000
>>>>>>> drl_multiprocessing
    else:
        n_timesteps = args.n

    # start training
    start = time.time()
    # print(start)
    model.learn(
        total_timesteps = n_timesteps, callback=eval_cb, reset_num_timesteps=True)
    print(f'Time passed for {n_timesteps} timesteps: {time.time()-start}s')

    # update the timesteps the model has trained in total
    update_total_timesteps_json(hyperparams_obj, n_timesteps, PATHS)
    print("training done!")
    