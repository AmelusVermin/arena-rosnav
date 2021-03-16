import os
import rospy

from datetime import datetime as dt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from task_generator.task_generator.tasks import get_predefined_task
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.scripts.custom_policy import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.rl_agent.envs.wp3_env import wp3Env
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.argsparser import parse_training_args
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.train_agent_utils import *
from arena_navigation.arena_local_planner.learning_based.arena_local_planner_drl.tools.custom_mlp_utils import *


###HYPERPARAMETER###
robot = "myrobot"
gamma = 0.99
n_steps = 500
ent_coef = 0.01
learning_rate = 2.5e-4
vf_coef = 0.5
max_grad_norm = 0.5
gae_lambda = 0.95
batch_size = 64
n_epochs = 4
clip_range = 0.2
reward_fnc = "00"
discrete_action_space = False
####################


class agent_hyperparams(object):
    """ Class containing agent specific hyperparameters (for documentation purposes)

    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param robot: Robot name to load robot specific .yaml file containing settings
    :param gamma: Discount factor
    :param n_steps: The number of steps to run for each environment per update
    :param ent_coef: Entropy coefficient for the loss calculation
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param reward_fnc: Number of the reward function (defined in ../rl_agent/utils/reward.py)
    :param discrete_action_space: If robot uses discrete action space
    :param n_timesteps: The number of timesteps trained on in total.
    """
    def __init__(self, agent_name: str, robot: str, gamma: float, n_steps: int, ent_coef: float, learning_rate: float, vf_coef: float, max_grad_norm: float, gae_lambda: float,
                 batch_size: int, n_epochs: int, clip_range: float, reward_fnc, discrete_action_space: bool, n_timesteps: int = 0):
        self.agent_name = agent_name
        self.robot = robot 
        self.gamma = gamma 
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.reward_fnc = reward_fnc
        self.discrete_action_space = discrete_action_space
        self.n_timesteps = n_timesteps


def get_agent_name(args):
    """ Function to get agent name to save to/load from file system
    
    Example names:
    "MLP_B_64-64_P_32-32_V_32-32_relu_2021_01_07__10_32"
    "DRL_LOCAL_PLANNER_2021_01_08__7_14"

    :param args (argparse.Namespace): Object containing the program arguments
    """
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

    if args.custom_mlp:
        return "MLP_B_" + args.body + "_P_" + args.pi + "_V_" + args.vf + "_" + args.act_fn + "_" + START_TIME
    if args.load is None:
        return args.agent + "_" + START_TIME
    return args.load


def get_paths(agent_name: str, args):
    """ Function to generate agent specific paths 
    
    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path('arena_local_planner_drl')

    PATHS = {
        'model' : os.path.join(dir, 'agents', agent_name),
        'tb' : os.path.join(dir, 'training_logs', 'tensorboard', agent_name),
        'eval' : os.path.join(dir, 'training_logs', 'train_eval_log', agent_name),
        'robot_setting' : os.path.join(rospkg.RosPack().get_path('simulator_setup'), 'robot', robot + '.model.yaml'),
        'robot_as' : os.path.join(rospkg.RosPack().get_path('arena_local_planner_drl'), 'configs', 'default_settings.yaml'),
    }

    hyperparams = agent_hyperparams(agent_name, robot, gamma, n_steps, ent_coef, learning_rate, vf_coef,max_grad_norm, gae_lambda, batch_size, 
                                    n_epochs, clip_range, reward_fnc, discrete_action_space)

    # check for mode
    if args.load is None:
        os.makedirs(PATHS.get('model'))
        write_hyperparameters_json(hyperparams, PATHS)
    else:
        if not os.path.isfile(os.path.join(PATHS.get('model'), AGENT_NAME + ".zip")) and not os.path.isfile(os.path.join(PATHS.get('model'), "best_model.zip")):
            raise FileNotFoundError("Couldn't find model named %s.zip' or 'best_model.zip' in '%s'" % (AGENT_NAME, PATHS.get('model')))

    #print_hyperparameters_json(hyperparams, PATHS)
    
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


if __name__ == "__main__":
    args, _ = parse_training_args()

    rospy.init_node("test", disable_signals=True)

    # generate agent name and model specific paths
    AGENT_NAME = get_agent_name(args)
    print("________ STARTING TRAINING WITH:  %s ________\n" % AGENT_NAME)
    PATHS = get_paths(AGENT_NAME, args)

    # set num of timesteps to be generated 
    if args.n is None:
        n_timesteps = 60000000
    else:
        n_timesteps = args.n

    # instantiate gym environment
    n_envs = 1
    task = get_predefined_task("random")
    env = DummyVecEnv([lambda: wp3Env(task, PATHS.get('robot_setting'), PATHS.get('robot_as'), discrete_action_space, goal_radius=1.25, max_steps_per_episode=550)] * n_envs)
   
    # instantiate eval environment
    eval_env = Monitor(wp3Env(task, PATHS.get('robot_setting'), PATHS.get('robot_as'), discrete_action_space, goal_radius=1.25, max_steps_per_episode=550), PATHS.get('eval'), info_keywords=("done_reason",))
    eval_cb = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=5000, log_path=PATHS.get('eval'), best_model_save_path=PATHS.get('model'), deterministic=True)

    # determine mode
    if args.custom_mlp:
        # custom mlp flag
        model = PPO("MlpPolicy", env, policy_kwargs = dict(net_arch = args.net_arch, activation_fn = get_act_fn(args.act_fn)), 
                    gamma = gamma, n_steps = n_steps, ent_coef = ent_coef, learning_rate = learning_rate, vf_coef = vf_coef, 
                    max_grad_norm = max_grad_norm, gae_lambda = gae_lambda, batch_size = batch_size, n_epochs = n_epochs, clip_range = clip_range, 
                    tensorboard_log = PATHS.get('tb'), verbose = 1)

    elif args.agent is not None:
        # predefined agent flag
        if args.agent == "MLP_ARENA2D":
                model = PPO(MLP_ARENA2D_POLICY, env, gamma = gamma, n_steps = n_steps, ent_coef = ent_coef, 
                        learning_rate = learning_rate, vf_coef = vf_coef, max_grad_norm = max_grad_norm, gae_lambda = gae_lambda, 
                        batch_size = batch_size, n_epochs = n_epochs, clip_range = clip_range, tensorboard_log = PATHS.get('tb'), verbose = 1)

        elif args.agent == "DRL_LOCAL_PLANNER" or args.agent == "CNN_NAVREP":
            if args.agent == "DRL_LOCAL_PLANNER":
                policy_kwargs = policy_kwargs_drl_local_planner
            else:
                policy_kwargs = policy_kwargs_navrep

            model = PPO("CnnPolicy", env, policy_kwargs = policy_kwargs, 
                gamma = gamma, n_steps = n_steps, ent_coef = ent_coef, learning_rate = learning_rate, vf_coef = vf_coef, 
                max_grad_norm = max_grad_norm, gae_lambda = gae_lambda, batch_size = batch_size, n_epochs = n_epochs, 
                clip_range = clip_range, tensorboard_log = PATHS.get('tb'), verbose = 1)
    
    else:
        # load flag
        if os.path.isfile(os.path.join(PATHS.get('model'), AGENT_NAME + ".zip")):
            model = PPO.load(os.path.join(PATHS.get('model'), AGENT_NAME), env)
        elif os.path.isfile(os.path.join(PATHS.get('model'), "best_model.zip")):
            model = PPO.load(os.path.join(PATHS.get('model'), "best_model"), env)

    # start training
    model.learn(total_timesteps = n_timesteps, callback=eval_cb, reset_num_timesteps = False)
    #model.save(os.path.join(PATHS.get('model'), AGENT_NAME))

    # update the timesteps the model has trained in total
    update_total_timesteps_json(n_timesteps, PATHS)

    print("training done and model saved!")
    
"""
    s = time.time()
    model.learn(total_timesteps = 3000)
    print("steps per second: {}".format(1000 / (time.time() - s)))
    # obs = env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic = True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #       obs = env.reset()
"""