from gym.spaces import space
import numpy as np
from tonian.tasks.base.command import Command
from tonian.tasks.base.vec_task import VecTask, BaseEnv, GenerationalVecTask


from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace

from gym import spaces
import gym

from typing import Dict, Any, Tuple, Union, Optional

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import yaml
import time
import os


import torch
from torch import nn 
import torch.nn as nn

class Mk1WalkingTask(GenerationalVecTask):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
            
    def _extract_params_from_config(self) -> None:
        return super()._extract_params_from_config()
    
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
    
    def _create_envs(self,spacing: float, num_per_row: int) -> None:
        
        pass
        
    
    def pre_physics_step(self, actions: torch.Tensor):
        return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        return super().post_physics_step()
    
    def reset_envs(env_ids: torch.Tensor) -> None:
        return super().reset_envs()
    
    def _is_symmetric(self):
        return False
    
    
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_actor_obs = 103
        return MultiSpace({
            "linear": spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, ))
        })
        
    def _get_critic_observation_spaces(self) -> MultiSpace:
        """Define the different observations for the critic of the agent
        
        
         The observations will later be combined with other inputs like commands to create the critic input space
        
        This is an asymmetric actor critic implementation  -> The critic observations differ from the actor observations
        and unlike the actor inputs the actor inputs don't have to be things that a real life robot could also observe in inference.
        
        Things like distance to target position, that can not be observed on site can be included in the critic input
    
        Returns:
            MultiSpace: [description]
        """
        num_critic_obs = 134
        return MultiSpace({
            "linear": spaces.Box(low=-1.0, high=1.0, shape=(num_critic_obs, ))
        })
    
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        num_actions = 21
        return spaces.Box(low=-1.0, high=1.0, shape=(num_actions, )) 
    
    def reward_range(self):
        return (-1e100, 1e100)
    
    
    def close(self):
        pass
    
    