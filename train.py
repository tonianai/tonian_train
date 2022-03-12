
from tonian.tasks.cartpole.cartpole_task import Cartpole

from typing import Dict, Union, List

import yaml

import sys

from tonian.common.utils.utils import set_random_seed
from tonian.common.utils.config_utils import task_from_config, algo_from_config, policy_from_config

import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn

import yaml

import argparse

device = "cuda:0"


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, help="Seed for running the env")
    ap.add_argument("-cfg", required= True, help="path to the config")
    
    args = vars(ap.parse_args())
    
    args_seed = None
    
    if args['seed'] is not None:
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
        config["algo"]["seed"] = args_seed
    elif "seed" in config:   
        set_random_seed(config["seed"])
    
    # create the run folder here and not in the 
    
    
    task = task_from_config(config["task"])
    task.is_symmetric = False
    policy = policy_from_config(config["policy"], task)
    print(policy)
    algo = algo_from_config(config["algo"], task, policy, device)
    
    algo.learn(total_timesteps=1e10)
    
    task.close()
    
    
    
        
