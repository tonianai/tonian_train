from typing import Dict, Union, List

import yaml

import sys

from tonian.tasks.walking.walking_task import WalkingTask
from tonian.tasks.cartpole.cartpole_task import Cartpole

import numpy as np

from tonian.algorithms.base_algorithm import BaseAlgorithm
from tonian.algorithms.ppo import PPO
from tonian.policies.policies import SimpleActorCriticPolicy, ActorCriticPolicy

from gym.spaces import space
from tonian.tasks.base.command import Command
from tonian.tasks.base.vec_task import MultiSpace, VecTask
from tonian.common.schedule import Schedule
from tonian.common.utils.utils import set_random_seed

import gym
from gym import spaces
import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole

import torch
import torch.nn as nn

import yaml

import argparse

device = "cuda:0"

def parseActvationFunction(string: str):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    return activations[string]


def task_from_config(config: Dict, headless: bool = False) -> VecTask:
    
    name_to_task_map=  {
     "00_cartpole": Cartpole,
     "01_walking": WalkingTask
    }
    return name_to_task_map[config["name"]](config, sim_device="gpu", graphics_device_id=0, headless= headless)
    
def policy_from_config(config: Dict, env: VecTask) -> ActorCriticPolicy:
    
    actor_obs_spaces = env.actor_observation_spaces
    critic_obs_spaces = env.critic_observation_spaces
    action_space = env.action_space
    
    assert "lr" in config.keys(), "The learning rate must be specified in the config file"
    
    
    lr = Schedule(config["lr"])
    if config['name'] == "SimpleActorCritic":
        
        activation_fn = nn.Tanh
        if "activation_fn" in config:
            activation_fn = parseActvationFunction(config["activation_fn"])
        
        if "actor_hidden_layers" in config:    
            actor_hidden_layers = tuple(config['actor_hidden_layers'])
        else:
            actor_hidden_layers = (128, 128, 128)
        
        if "critic_hidden_layers" in config:    
            critic_hidden_layers = tuple(config['critic_hidden_layers'])
        else:
            critic_hidden_layers = (128,128,128)
            
        if "device" in config:
            device = config["device"]
        else:
            device = "cuda:0"
             
            
        # create a simple actor critic from the params
        return SimpleActorCriticPolicy(actor_obs_space= actor_obs_spaces,
                                       critic_obs_space= critic_obs_spaces,
                                       lr_schedule= lr,
                                       action_space=action_space,
                                       activation_fn_class=activation_fn,
                                       actor_hidden_layer_sizes=actor_hidden_layers,
                                       critic_hiddent_layer_sizes=critic_hidden_layers,
                                       device= device
                                       )
        # TODO: Make more stuff configurable
    else:
        raise Exception("Not supported Policy name from Configuration File")    
    
    pass

def algo_from_config(config: Dict, env: VecTask, policy: ActorCriticPolicy, device: str) -> BaseAlgorithm:
    
    if config['name'] == 'PPO':
        return PPO(env, config, policy, device)
    else:
        raise Exception("Not supported Alogorithm name from Configuration File")
    
    pass


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, help="Seed for running the env")
    ap.add_argument("-cfg", required= True, help="path to the config")
    
    args = vars(ap.parse_args())
    
    args_seed = int(args['seed'])
    
    device = "cuda:0"

    config_path = args['cfg']
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
    
    if args_seed is not None:
        set_random_seed(args_seed)
    elif "seed" in config:   
        set_random_seed(config["seed"])
    
    
    
    task = task_from_config(config["task"])
    task.is_symmetric = False
    policy = policy_from_config(config["policy"], task)
    print(policy)
    algo = algo_from_config(config["algo"], task, policy, device)
    
    algo.learn(total_timesteps=1e10)
    
    task.close()
    
    
    
        
