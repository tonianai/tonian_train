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

import gym
from gym import spaces
import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole

import torch

import yaml

device = "cuda:0"

def task_from_config(config: Dict, headless: bool = False) -> VecTask:
    
    name_to_task_map=  {
     "00_cartpole": Cartpole,
     "01_walking": WalkingTask
    }
    return name_to_task_map[config["name"]](config, sim_device="gpu", graphics_device_id=0, headless= headless)
    

def algo_from_config(config: Dict) -> BaseAlgorithm:
    
    pass

def policy_from_config(config: Dict) -> ActorCriticPolicy:
    pass


if __name__ == "__main__":
    
    assert len(sys.argv) == 2, "Usage python3 train.py <path_to_config>"
        
    config_path = sys.argv[1]
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
    
    task = task_from_config(config["task"])
    
    print(task)
    
    
    
    
        
