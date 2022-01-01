from gym.spaces import space
import numpy as np
from elysium.tasks.base.command import Command
from elysium.tasks.base.vec_task import VecTask, BaseEnv, GenerationalVecTask


from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from elysium.common.utils.spaces import MultiSpace

from gym import spaces
import gym

from typing import Dict, Any, Tuple, Union

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import yaml
import time
import os


import torch
from torch import nn 
import torch.nn as nn

class WalkingTask(GenerationalVecTask):
    
    
    
    def __init__(self, config_path: str, sim_device: str, graphics_device_id: int, headless: bool) -> None:
        
        
        config = self._fetch_config_params(config_path)
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    

        self._get_gpu_gym_state_tensors()
        
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        
        
        self.heading_vector = to_torch([1,0,0], device= self.device).repeat((self.num_envs, 1))
        self.initial_heading_vector = self.heading_vector.clone()
        
    def _fetch_config_params(self, config_path: str) -> Dict:
        """Fetches the config file and extracts config properties from it and sets important member variables
        Args:
            config_path (str): relative path to the config file
        """
        
        # opfen the config file 
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"File {config_path} not found")
        
        #extract params from config 
        self.randomize = config["task"]["randomize"]
        
        self.power_scale = config["env"]["powerscale"]
        
        
        self.dt = config["sim"]["dt"]
        
        # The reward weighting dict is located in the config.yaml file and determines how much each reward contributes to the total reward function
        self.reward_weighting = config["env"]["reward_weighting"]
        
        return config
            
    def _get_gpu_gym_state_tensors(self) -> None:
        """Retreive references to the gym tensors for the environment, that are on the gpu
        """
        
        # --- aquire tensor pointers
        # the state of each root body is represented using 13 floats with the same layout as GymRigidBodyState: 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) 
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
         
        
        # --- wrap pointers to torch tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # root states of the actors -> shape( numenvs, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        # Todo: Change this value when you add more sensors
        sensors_per_env = 2
        self.vec_force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        
        # --- refresh the tensors
        self.refresh_tensors()
        
        

        # positions of the joints
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # velocities of the joints
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # --- set initial tensors
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0
        
        
        pass
    
    def refresh_tensors(self):
        """Refreshes tensors, that are on the GPU        """
        self.gym.refresh_dof_state_tensor(self.sim) # state tensor contains velocities and position for the jonts 
        self.gym.refresh_actor_root_state_tensor(self.sim) # root state tensor contains ground truth information about the root link of the actor
        self.gym.refresh_force_sensor_tensor(self.sim) # the tensor of the added force sensors (added in _create_envs)
        self.gym.refresh_dof_force_tensor(self.sim) # dof force tensor contains foces applied to the joints
    
    def reset(self) -> Tuple[Dict[str, torch.Tensor]]:
        """Reset the environment and gather the first obs
        
        Returns:
            Tuple[Dict[str, torch.Tensor]]: actor_obs, critic_obs
        """
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
        """Compute Observations and Calculate reward"""
        
        # The data in the tensors must be correct, before the observation tensors can be computed
        self.refresh_tensors()
        
        # use jit script to compute the observations
        self.actor_obs, self.critic_obs = compute_robot_observations(
            root_states = self.root_states, 
            sensor_states=self.vec_force_sensor_tensor,
            dof_vel=self.dof_vel,
            dof_pos= self.dof_pos,
            dof_limits_lower=self.dof_limits_lower,
            dof_limits_upper=self.dof_limits_upper,
            dof_force= self.dof_force_tensor,
            initial_heading= self.initial_heading_vector,
            actions= self.actions
        )
        
        
        self.rewards , self.do_reset = compute_robot_rewards(
            root_states= self.root_states,
            sensor_states=self.vec_force_sensor_tensor,
            alive_reward= 0.1,
            death_cost= -1.0,
            directional_factor= 1.0,
            energy_cost= 1.0
        )
        
        
        
        
        
    def reset_envs(self, env_ids):
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
        
    def apply_randomizations(self):
        pass
    
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
        
        

@torch.jit.script
def compute_robot_rewards(root_states: torch.Tensor,
                          sensor_states: torch.Tensor,
                          death_cost: float,
                          alive_reward: float, 
                          directional_factor: float,
                          energy_cost: float
                          )-> Tuple[torch.Tensor, torch.Tensor]:
        
        
        #base reward for being alive  
        reward = torch.ones_like(root_states[:, 0]) * alive_reward
        
        # Todo: Add other values to reward function and make the robots acceptable to command
        
        # reward for proper heading
        
        # reward for being upright
        
        
        # punish for having fallen
        terminations_height = 0.8
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(root_states[:, 2] < terminations_height, torch.ones_like(reward) * death_cost, reward)
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
         
        # reward for power effinecy
        
        # reward for runnign speed 
        return (reward, has_fallen)

@torch.jit.script
def compute_robot_observations(root_states: torch.Tensor, 
                                sensor_states: torch.Tensor, 
                                dof_vel: torch.Tensor, 
                                dof_pos: torch.Tensor, 
                                dof_limits_lower: torch.Tensor,
                                dof_limits_upper: torch.Tensor,
                                dof_force: torch.Tensor,
                                actions: torch.Tensor,
                                initial_heading: torch.Tensor
                                ):
    
    
    """Calculate the observation tensors for the crititc and the actor for the humanoid robot
    
    Note: The resulting tensors must be in the same shape as the multispaces: 
        - self.actor_observation_spaces
        - self.critic_observatiom_spaces

    Args:
        root_states (torch.Tensor): Root states contain things like positions, velcocities, angular velocities and orientation of the root of the robot 
        sensor_states (torch.Tensor): state of the sensors given 
        dof_vel (torch.Tensor): velocity tensor of the dofs
        dof_pos (torch.Tensor): position tensor of the dofs
        dof_force (torch.Tensor): force tensor of the dofs
        actions (torch.Tensor): actions of the previous 

    Returns:
        Tuple[Dict[torch.Tensor]]: (actor observation tensor, critic observation tensor)
    """
    
    
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    
    
    # todo add some other code to deal with initial information, that might be required
    
    
    # todo: the actor still needs a couple of accelerometers
    linear_actor_obs = torch.cat((sensor_states, dof_pos, dof_vel, dof_force, ang_velocity, torso_rotation, actions), dim=-1)
    
    
    linear_critic_obs = torch.cat((linear_actor_obs, torso_rotation, velocity, torso_position, actions), dim=-1)
    
    return  {'linear': linear_actor_obs},  {'linear': linear_critic_obs}