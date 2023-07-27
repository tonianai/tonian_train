from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod, abstractproperty

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import numpy as np

import torch  
 

from tonian_train.common.spaces import MultiSpace


class BaseEnv(ABC):
    
    def __init__(self,config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str= "cuda:0") -> None:
        """An asymmetric actor critic base environment class based on isaac gym 

        Args:
            config (Dict[str, Any]): the config dictionary
            sim_device (str): ex: cuda:0, cuda or cpu
            graphics_device_id (int): The device id to render with
            headless (bool): determines whether the scene is rendered
        """
         
                
        self.config = config
        self.headless = headless
        
        
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0
 
        self.action_space = self._get_action_space()
        
        
        self.metadata = {}
        
    
    @property
    def actor_observation_spaces(self) -> gym.Space:
        return self._get_actor_observation_spaces()
        
    
    @property
    def observation_space(self) -> gym.Space:
        return self.actor_observation_spaces
    

    @abstractproperty
    def reward_range(self):
        pass
        
        
        
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any], Optional[Dict[str, float]]]:
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]], Optional[Dict[str, float]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info, reward_constituents
        """
        
        
        
        pass
    
    @abstractmethod
    def reset(self) -> Tuple[Dict[str, torch.Tensor]]:
        """Reset the complete environment and return a output multispace
        Returns:
            Dict[str, torch.Tensor]: Output multispace (names in the dict correspond to those given in the multispace),
        """
        pass



    
    @abstractmethod
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different inputs the actor of the agent
         (this includes linear observations, viusal observations)
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference
        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()
    
    
    
    @abstractmethod
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment properly
        """
        