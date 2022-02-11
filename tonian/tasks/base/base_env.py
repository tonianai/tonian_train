from typing import Dict, Any, Union, Tuple
from abc import ABC, abstractmethod, abstractproperty

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import numpy as np

import torch 
import torch.nn as nn

import time

import sys

from tonian.common.spaces import MultiSpace


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

        
        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", rl_device)
        
        
        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:   
            self.graphics_device_id = -1
        
        self.num_envs = config["env"]["num_envs"] 

        # The Frequency with which the actions are polled relative to physics step
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.clip_obs = config["env"].get("clip_observations", np.Inf)
        self.clip_actions = config["env"].get("clip_actions", np.Inf)
        
        
        # This implementation used Asymmetic Actor Critics
        # https://arxiv.org/abs/1710.06542
        self.critic_observation_spaces = self._get_critic_observation_spaces()
        self.actor_observation_spaces = self._get_actor_observation_spaces()
        self.action_space = self._get_action_space()
        
        
        self.metadata = {}
        
    
    @property
    def observation_space(self) -> gym.Space:
        return spaces.Dict(self.actor_observation_spaces.spaces)

    @abstractproperty
    def reward_range(self):
        pass
        
        
        
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info
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
    def _get_critic_observation_spaces(self) -> MultiSpace:
        """Define the different inputs for the critic of the agent
        
        This is an asymmetric actor critic implementation  -> The critic observations differ from the actor observations
        and unlike the actor inputs the actor inputs don't have to be things that a real life robot could also observe in inference.
        
        Things like distance to target position, that can not be observed on site can be included in the critic input
    
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