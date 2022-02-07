import argparse
import os
import argparse
from random import seed
import gym
from typing import Type
from stable_baselines import PPO2
from stable_baselines3.common.base_class import BaseAlgorithm

from .nn_utils import get_act_fn, get_net_arch
from policies.sb_policy_registry import PolicyRegistry
# a dict containing all cases, where the agents are built the same way


class ModelBuilder:
    
    @staticmethod
    def get_model(args: argparse.Namespace, save_paths: dict, env: Type[gym.Env]) -> Type[BaseAlgorithm]: 
        """ 
        build or load stable_baseline3 models.

        Parameters:
            args (argparse.Namespace): contains all command line arguments and config parameters
            model_params (argparse.Namespace): contains model specific params
            save_paths (dict) : should contain a path for tensorboard. e.g. save_paths["tensorboard"] = "a/path/for/logging"
            env (Type[gym.Env]) : the environment based on gym.Env used for training
        Returns:
            model (Type[BaseAlgorithm]) : an algorithm object from stable_baselines3 like PPO representing the model 
        """
        if not args.load_model:
            # build model according to CUSTOM_MLP
            if args.agent_type == "CUSTOM_MLP":    
                # get network architecture
                net_arch = get_net_arch(args)
                model = PPO2(
                        "MlpPolicy",
                        env,
                        policy_kwargs=dict(net_arch=net_arch, act_fun=get_act_fn(args.act_fn)),
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        lam=args.gae_lambda,
                        nminibatches=args.mini_batch_size,
                        noptepochs=args.n_epochs,
                        cliprange=args.clip_range,
                        seed=args.seed,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                        n_cpu_tf_sess=args.n_cpu_tf_sess
                    )
            elif args.agent_type == "CUSTOM_MLP_LN_LSTM":
                # get network architecture
                net_arch = get_net_arch(args, is_lstm=True)
                model = PPO2(
                        "MlpLnLstmPolicy",
                        env,
                        policy_kwargs=dict(net_arch=net_arch, act_fun=get_act_fn(args.act_fn), n_lstm=args.number_lstm_cells),
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        lam=args.gae_lambda,
                        nminibatches=args.mini_batch_size,
                        noptepochs=args.n_epochs,
                        cliprange=args.clip_range,
                        seed=args.seed,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                        n_cpu_tf_sess=args.n_cpu_tf_sess
                    )
            # build a Base Agent PPO model
            elif args.agent_type in PolicyRegistry.get_all_registered_policies():
                model = PPO2(
                        args.agent_type,
                        env,
                        policy_kwargs=dict(),
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        lam=args.gae_lambda,
                        nminibatches=args.mini_batch_size,
                        noptepochs=args.n_epochs,
                        cliprange=args.clip_range,
                        seed=args.seed,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                        n_cpu_tf_sess=args.n_cpu_tf_sess
                    )
            else :
                raise NameError(f"Agent type {args.agent_type} is not specified for building!")
            print("build new model")
        else:
            # load model from zip file if it exists
            if os.path.isfile(args.model_path):
                model = PPO2.load(os.path.join(args.model_path), env)
                # the paths need to updated as set one might not exist
                model.tensorboard_log = save_paths["tensorboard"]
                print(f"loaded model: {args.model_path}")
                # overwrite params if wished
                if args.overwrite_params:
                    ModelBuilder.overwrite_model_hyperparams(model, args)
            else:
                raise FileNotFoundError(f"{args.model_path} is not a file!")

        return model
    
    @staticmethod
    def overwrite_model_hyperparams(model: PPO2, args: argparse.Namespace):
        """
        Updates parameter of loaded PPO agent when it was manually changed in the configs yaml.

        Parameters:
            model (PPO): loaded PPO agent
            args (argparse.Namespace): conatins training related parameters and the number of environments
        """
        
        model.n_batch = args.n_steps * args.n_envs
        model.gamma = args.gamma
        model.n_steps = args.n_steps
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.vf_coef = args.vf_coef
        model.max_grad_norm = args.max_grad_norm
        model.lam = args.gae_lambda
        model.noptepochs = args.n_epochs
        model.nminibatches=args.mini_batch_size