import argparse
import os
import argparse
import gym
from typing import Type, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.base_class import BaseAlgorithm

from .nn_utils import get_act_fn, get_net_arch
from model.base_agent import BaseAgent, PolicyType
from model.agent_factory import AgentFactory

# a dict containing all cases, where the agents are built the same way
AGENT_CASES = {
    "CustomMLPPPO" : ["CUSTOM_MLP"],
    "BaseAgentsPPO" : ["AGENT_6", "AGENT_7", "AGENT_8", "AGENT_9", "AGENT_10", "AGENT_11", "AGENT_17", 
            "AGENT_18", "AGENT_19", "AGENT_20", "AGENT_21", "AGENT_22", "AGENT_23"],
    "ActorCriticPPO" : "MLP_ARENA2D"
}

class ModelBuilder:
    
    @staticmethod
    def get_model(args: argparse.Namespace, model_params: argparse.Namespace, save_paths: dict, env: Type[gym.Env]) -> Type[BaseAlgorithm]: 
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
            if args.agent_type in AGENT_CASES["CustomMLPPPO"]:
                # model_params should contain body, pi, vf, act_fn for this case
                attributes = ["body", "pi", "vf", "act_fn"]
                assert(
                    all(hasattr(model_params, att) for att in attributes), 
                    f"one of the following attributes are missing in the config file {args.model_config_path}: {attributes}"
                )
                # get network architecture
                net_arch = get_net_arch(model_params)
                model = PPO(
                        PolicyType.MLP,
                        env,
                        policy_kwargs=dict(net_arch=net_arch, activation_fn=get_act_fn(model_params.act_fn)),
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        gae_lambda=args.gae_lambda,
                        batch_size=args.mini_batch_size,
                        n_epochs=args.n_epochs,
                        clip_range=args.clip_range,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                    )
            # build a Base Agent PPO model
            elif args.agent_type in AGENT_CASES["BaseAgentsPPO"]:
                agent: Union[Type[BaseAgent], Type[ActorCriticPolicy]] = AgentFactory.instantiate(args.agent_type)
                model = PPO(
                        agent.type.value,
                        env,
                        policy_kwargs=agent.get_kwargs(),
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        gae_lambda=args.gae_lambda,
                        batch_size=args.mini_batch_size,
                        n_epochs=args.n_epochs,
                        clip_range=args.clip_range,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                    )
            # build a custom ActorCritic PPO model
            elif args.agent_type in AGENT_CASES["ActorCriticPPO"]:
                agent: Union[Type[BaseAgent], Type[ActorCriticPolicy]] = AgentFactory.instantiate(args.agent_type)
                model = PPO(
                        agent,
                        env,
                        gamma=args.gamma,
                        n_steps=args.n_steps,
                        ent_coef=args.ent_coef,
                        learning_rate=args.learning_rate,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        gae_lambda=args.gae_lambda,
                        batch_size=args.mini_batch_size,
                        n_epochs=args.n_epochs,
                        clip_range=args.clip_range,
                        tensorboard_log=save_paths['tensorboard'],
                        verbose=args.train_verbose,
                    )
            else :
                raise NameError(f"Agent type {args.agent_type} is not specified for building!")
        else:
            # load model from zip file if it exists
            if os.path.isfile(args.model_path):
                model = PPO.load(os.path.join(args.model_path), env)
                # the paths need to updated as set one might not exist
                model.tensorboard_log = save_paths["tensorboard"]
                # overwrite params if wished
                if args.overwrite_params:
                    ModelBuilder.overwrite_model_hyperparams(model, args, save_paths, args.n_envs)
            else:
                raise FileNotFoundError(f"{args.model_path} is not a file!")

        return model
    
    @staticmethod
    def overwrite_model_hyperparams(model: PPO, args: argparse.Namespace):
        """
        Updates parameter of loaded PPO agent when it was manually changed in the configs yaml.

        Parameters:
            model (PPO): loaded PPO agent
            args (argparse.Namespace): conatins training related parameters and the number of environments
            n_envs (int): number of parallel environments
        """
        
        model.batch_size = args.batch_size
        model.gamma = args.gamma
        model.n_steps = args.n_steps
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.vf_coef = args.vf_coef
        model.max_grad_norm = args.max_grad_norm
        model.gae_lambda = args.gae_lambda
        model.n_epochs = args.n_epochs
        """
        if model.clip_range != params['clip_range']:
            model.clip_range = params['clip_range']
        """
        if model.n_envs != args.n_envs:
            model.update_n_envs() 
        model.rollout_buffer.buffer_size = args.n_steps