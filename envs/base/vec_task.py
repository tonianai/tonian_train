from typing import Dict, Any, Union, Tuple
from abc import ABC, abstractmethod, abstractproperty

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import numpy as np

import torch 
import torch.nn as nn



class MultiSpace():
    """
    Defines the combination of multiple spaces to one multispace
    
    For expample an agent might require linear observations, visual observations and command input
    or an agent might output communication to other agents as well as actions
    These spaces can be summarized within multispace
    """
    def __init__(self, spaces: Dict[str, gym.Space] ) -> None:
        
        self.spaces = spaces
        self.space_names = spaces.keys()
        self.num_spaces = len(spaces)
        self.shape = tuple([self.spaces[i].shape for i in self.spaces])
        
    
    def __len__(self):
        return self.num_spaces         

    def __str__(self):
        return f"Multispace: \n Shape: {str(self.shape)} \n Contains: {str(self.spaces)}"
        
class Env(ABC):
    
    def __init__(self,config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool) -> None:
        """[summary]

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

        self.rl_device = config.get("rl_device", "cuda:0")
        
        
        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:   
            self.graphics_device_id = -1
        
        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)
        
        
        # Input spaces is the multi Space pardon to the 
        self.input_spaces = self._get_input_spaces()
        
        self.output_spaces = self._get_output_spaces()
        
        
        
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
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the complete environment and return a output multispace
        Returns:
            Dict[str, torch.Tensor]: Output multispace (names in the dict correspond to those given in the multispace),
        """
        pass
        
    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments
    
    @abstractmethod
    def _get_input_spaces(self) -> MultiSpace:
        pass
    
    @abstractmethod
    def _get_output_spaces(self) -> MultiSpace:
        pass 

class VecTask(Env, ABC):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool) -> None:
        """Initialise the `VecTask`.

        Args:
            config (Dict[str, Any]): the config dictionary
            sim_device (str): ex: cuda:0, cuda or cpu
            graphics_device_id (int): The device id to render with
            headless (bool): determines whether the scene is rendered
        """
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        
        self.sim_params = self.__parse_sim_params(self.config["physics_engine"], self.config["sim"])
        if self.config["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.config["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.config['physics_engine']}"
            raise ValueError(msg)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params
