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
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
            
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
        
        assert self.config["sim"] is not None, "The sim config must be set on the task config file"
        assert self.config["env"] is not None, "The env config must be set on the task config file"
        
        
        
        reward_weight_dict = self.config["env"]["reward_weighting"]  
        
        
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
    
    def _get_gpu_gym_state_tensors(self) -> None:
        """
        Retreive references to the gym tensors for the enviroment, that are on the gpu
        """
        # --- aquire tensor pointers
        # the state of each root body is represented using 13 floats with the same layout as GymRigidBodyState: 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) 
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
      
        # --- wrap pointers to torch tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        
    
    
    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)


        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf/mk-1/")
        
        mk1_robot_file = "robot.urdf"
        
        asset_options = gymapi.AssetOptions()
        
        asset_options.fix_base_link = False
        
        mk1_robot_asset = self.gym.load_asset(self.sim, asset_root, mk1_robot_file, asset_options)
        
        self.num_dof = self.gym.get_asset_dof_count(mk1_robot_asset)
        
        pose = gymapi.Transform()
        
        self.robot_handles = []
        self.envs = [] 
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, mk1_robot_asset, pose, "mk1", i, 1, 0)
            
            dof_probs = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            
            # TODO: Maybe change dof properties
            #print(dof_probs)
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            

            
        
    
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
    
    