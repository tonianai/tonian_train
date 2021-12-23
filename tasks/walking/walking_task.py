from gym.spaces import space
import torch
from torch import nn 
import torch.nn as nn
import numpy as np
from tasks.base.vec_task import VecTask, Env

from common.spaces import MultiSpace

from gym import spaces
import gym

from typing import Dict, Any, Tuple, Union

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import yaml
import time
import os

class WalkingTask(VecTask):
    
    def __init__(self, config_path: str, sim_device: str, graphics_device_id: int, headless: bool) -> None:
         
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"File {config_path} not found")
      
        #extract params from config 
        self.randomize = config["task"]["randomize"]
        
        self.power_scale = config["env"]["powerscale"]
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_obs = 30
        command_size = 10
        return MultiSpace({
            "linear": spaces.Box(low=-1.0, high=1.0, shape=(num_obs, )),
            "command": spaces.Box(low=-1.0, high = 1.0, shape=(command_size, ))
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
        return self._get_actor_observation_spaces()
    
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        num_actions = 21
        return spaces.Box(low=-1.0, high=1.0, shape=(num_actions, )) 
    
    
    def reset(self) -> Dict[str, torch.Tensor]:
        return super().reset()
    
    

    
    def pre_physics_step(self, actions: torch.Tensor):
        """Appl the action given to all the envs
        Args:
            actions (torch.Tensor): Expected Shape (num_envs, ) + self._get_action_space.shape

        Returns:
            [type]: [description]
        """
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        
        
        
        return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        
        
        return super().post_physics_step()

    def compute_rewards(self):
        
        # reward for proper heading
        
        # reward for being upright
        
        # punish for having fallen
        
        # reward for correct running speed
        
        # reward for 
                
        pass
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        super().step(actions)
    
    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        
        print(f"Create envs num_envs={self.num_envs} spacing = {spacing}, num_per_row={num_per_row}")
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "samples/nv_humanoid.xml"
        #asset_file = "urdf/mk-1/robot.urdf"
        
        if "asset" in self.config["env"]:
            asset_file = self.config["env"]["asset"].get("assetFileName", asset_file)
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        
        start_pose = gymapi.Transform()    
        start_pose.p = gymapi.Vec3(0.0,0.0, 1.35)
    
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        
        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(robot_asset)
        self.motor_efforts = to_torch([prop.motor_effort for prop in actuator_props])
        
        
        self.envs = []
        
        for i in range(self.num_envs):
            # create an env instance
            env_pointer = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            handle = self.gym.create_actor(env_pointer, robot_asset, start_pose, "robot", i, 0,0)
            
            self.envs.append(env_pointer)
            
        
        
        
        