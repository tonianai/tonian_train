from gym.spaces import space
import torch
from torch import nn 
import torch.nn as nn
import numpy as np
from tasks.base.command import Command
from tasks.base.vec_task import VecTask, Env


from isaacgym.torch_utils import torch_rand_float, tensor_clamp

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
    
    
         # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # root states of the actors -> shape( numenvs, 13)
        
        
        self.initial_root_states = self.root_states.clone()
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        
        
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.config["sim"]["dt"]
        
        
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
        return MultiSpace({
            "linear": spaces.Box(low=-1.0, high=1.0, shape=(num_obs, ))
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
        
        
        
    def post_physics_step(self):
        
        self.compute_observations()
        
        print(self.dof_pos.shape)
    
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim) # refreseh 
        self.gym.refresh_actor_root_state_tensor(self.sim) # refreshes the self.root_states tensor
        self.gym.refresh_rigid_body_state_tensor(self.sim)  #THIS!
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim) # 
        
        
        
        

    def reset_actor(self, env_ids):
        # Randomization can only happen at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations()
        
        
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)    
        
        
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        
        
    def apply_randomizations(self):
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
        
        # get ids used to attach force sensors at the feet
        # TODO: chnage for the humanoid robot
        
        
        sensor_pose = gymapi.Transform()
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(robot_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(robot_asset, "left_foot")
        
        self.gym.create_asset_force_sensor(robot_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(robot_asset, left_foot_idx, sensor_pose)
        
        
        
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.envs = []
        
        for i in range(self.num_envs):
            # create an env instance
            env_pointer = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            handle = self.gym.create_actor(env_pointer, robot_asset, start_pose, "robot", i, 0,0)
            
            self.gym.enable_actor_dof_force_sensors(env_pointer, handle)
            
            self.envs.append(env_pointer)
            
        
        dof_prop = self.gym.get_actor_dof_properties(env_pointer, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)
        


def compute_robot_rewards():
        
        # reward for proper heading
        
        # reward for being upright
        
        # punish for having fallen
        
        # reward for power effinecy
        
        # reward for runnign speed 
                
        pass
    
def compute_critic_observations(root_states, sensor_statems, dof_vel, dof_pos ):
    
    
    pass


def compute_actor_observations(sensor_states, dof_vel, dof_pos):
    
    pass