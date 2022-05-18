
from abc import ABC, abstractmethod

from gym.spaces import space
import numpy as np 
from tonian.tasks.base.vec_task import VecTask
from tonian.tasks.mk1.mk1_base import Mk1BaseClass
from tonian.tasks.common.task_dists import task_dist_from_config
from tonian.common.utils import join_configs

from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace
from tonian.common.torch_jit_utils import batch_dot_product

from tonian.tasks.common.task_dists import sample_tensor_dist

 

from typing import Dict, Any, Tuple, Union, Optional, List

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import os, torch, gym, yaml, time

class Mk1ControlledTerrainTask(Mk1BaseClass):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:
        
        base_config = self._get_standard_config()
        
        config = join_configs(base_config, config)
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        super()._create_envs(spacing, num_per_row) 
        
        
        self.upper_body_joint_names = ['left_shoulder_a', 
                                       'right_shoulder_a',
                                       'left_shoulder_b',
                                       'right_shoulder_b',
                                       'right_arm_rotate',
                                       'left_arm_rotate',
                                       'right_elbow',
                                       'left_elbow'] # all the names of the upper body joints, that for instance should be used minimally when walking
        self.upper_body_joint_indices = torch.LongTensor([ self.dof_name_index_dict[name] for name  in  self.upper_body_joint_names ])
         
        
        
        
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        super().reset_envs(env_ids)  
        
        n_envs_reset =  len(env_ids)
        
        self.target_velocity[env_ids] = sample_tensor_dist(self.target_velocity_dist, sample_shape=(n_envs_reset, ), device = self.device)
        
        self.target_direction[env_ids, 0] = torch.normal(mean = self.x_direction_mean[env_ids], std = self.x_direction_std[env_ids])
        self.target_direction[env_ids, 1] = torch.normal(mean = self.y_direction_mean[env_ids], std = self.y_direction_std[env_ids])
    
        
        
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
         
        assert self.config["mk1_controlled"] is not None, "The mk1_controlled config must be set on the task config file"
        
        reward_weight_dict = self.config["mk1_controlled"]["reward_weighting"]  
        
        self.energy_cost = reward_weight_dict["energy_cost"]
        self.death_cost = reward_weight_dict["death_cost"]
        self.alive_reward = float(reward_weight_dict["alive_reward"])
        self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        self.death_height = reward_weight_dict["death_height"]
        self.overextend_cost = reward_weight_dict["overextend_cost"]
        self.die_on_contact = reward_weight_dict.get("die_on_contact", True)
        self.contact_punishment_factor = reward_weight_dict["contact_punishment"]
        self.slowdown_punish_difference = reward_weight_dict["slowdown_punish_difference"]
        
        self.target_velocity_factor = reward_weight_dict["target_velocity_factor"]
        
        self.target_direction_factor = reward_weight_dict["target_direction_factor"]
        
        self.arm_use_cost = reward_weight_dict['arm_use_cost']

        controls_dict = self.config['mk1_controlled']['controls']

        self.target_velocity_dist =  controls_dict['velocity']
        self.target_x_dist = controls_dict['direction_x']
        self.target_y_dist = controls_dict['direction_y']
        
    def allocate_buffers(self):
        """Allocate all important tensors and buffers
        """
        super().allocate_buffers()
        
        self.x_direction_mean = torch.ones((self.num_envs, ), device=self.device) * self.target_x_dist['mean']
        self.x_direction_std = torch.ones((self.num_envs, ), device=self.device) * self.target_x_dist['std']
        x_direction = torch.normal(mean = self.x_direction_mean, std = self.x_direction_std).unsqueeze(dim=1)
        
        self.y_direction_mean = torch.ones((self.num_envs, ), device=self.device) * self.target_y_dist['mean']
        self.y_direction_std = torch.ones((self.num_envs, ), device=self.device) * self.target_y_dist['std']
        y_direction = torch.normal(mean = self.y_direction_mean, std = self.y_direction_std).unsqueeze(dim=1)
        
        
        
        self.target_direction =  torch.cat((x_direction, y_direction), dim= 1)
        
        self.target_velocity = sample_tensor_dist(self.target_velocity_dist, sample_shape=(self.num_envs, ), device= self.device)
        
        
    def refresh_tensors(self):
        super().refresh_tensors()
        
        self.gym.render_all_camera_sensors(self.sim)


    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor,]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
               Tuple[torch.Tensor, torch.Tensor]: 
                   reward: torch.Tensor shape(num_envs, )
                   has_fallen: torch.Tensor shape(num_envs, )
                   constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
        """
    
    
        # -------------- base reward for being alive --------------  
        
        reward = torch.ones_like(self.root_states[:, 0]) * self.alive_reward  
        
        
        quat_rotation = self.root_states[: , 3:7]
        
        #  -------------- reward for an upright torso -------------- 
        
        # The upright value ranges from 0 to 1, where 0 is completely horizontal and 1 is completely upright
        # Calulation explanation: 
        # take the first and the last value of the quaternion and take the quclidean distance
        upright_value = torch.sqrt(torch.sum( torch.square(quat_rotation[:, 0:4:3]), dim= 1 ))
        
        upright_punishment = (upright_value -1) * self.upright_punishment_factor
        
        reward += upright_punishment
        
        #  -------------- reward for speed in the right heading direction -------------- 
        
        linear_velocity_x_y = self.root_states[: , 7:9]
        
        # direction_in_deg base is -> neg x Axis
        direction_in_deg_to_x = torch.acos(quat_rotation[:, 0]) * 2
        
        # unit vecor of heading when seen from above 
        # this unit vector makes little sense, when the robot is highly non vertical
        two_d_heading_direction = torch.transpose(torch.cat((torch.unsqueeze(torch.sin(direction_in_deg_to_x), dim=0), torch.unsqueeze(torch.cos(direction_in_deg_to_x),dim=0) ), dim = 0), 0, 1)
        
        # compare the two_d_heading_direction with the linear_velocity_x_y using the angle between them
        # magnitude of the velocity (2 norm)
        vel_norm = torch.linalg.vector_norm(linear_velocity_x_y, dim=1)
        
        # positive is to fast and neg is to slow 
        vel_difference = vel_norm - self.target_velocity 
        
        vel_reward_factor = torch.where(vel_difference > 0 , compute_velocity_reward_factor(vel_difference, self.slowdown_punish_difference, self.target_velocity_factor), compute_velocity_reward_factor(- vel_difference, self.target_velocity, self.target_velocity_factor))
         
        #heading_to_velocity_angle = torch.arccos( torch.dot(two_d_heading_direction, linear_velocity_x_y)  / vel_norm )
        heading_to_velocity_angle = torch.arccos( batch_dot_product(two_d_heading_direction, linear_velocity_x_y) / vel_norm)
         
        
        target_velocity_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_reward_factor, torch.zeros_like(reward))
    
        reward += target_velocity_reward  
        
        # -------------- Punish for jittery motion (see ./research/2022-03-27_reduction-of-jittery-motion-in-action.md)--------------
        
        jitter_punishment = torch.abs(self.actions - self.former_actions).view(reward.shape[0], -1).sum(-1) * self.jitter_cost
        reward -= jitter_punishment
        
        
        
        #-------------- cost for overextension --------------
        distance_to_upper = self.dof_limits_upper - self.dof_pos
        distance_to_lower = self.dof_pos - self.dof_limits_lower
        distance_to_limit = torch.minimum(distance_to_upper, distance_to_lower)
        
        # 0.001 rad -> 0,071 deg 
        at_upper_limit = torch.where(distance_to_upper < 0.02, self.actions, torch.zeros_like(distance_to_limit))
        at_lower_limit = torch.where(distance_to_lower < 0.02, self.actions, torch.zeros_like(distance_to_lower)) * -1
        at_lower_limit[:, 8] = 0
        at_upper_limit[: , 8] = 0
        
        clipped_upper_punishment = torch.clamp(at_upper_limit, min=0) * self.overextend_cost
        clipped_lower_punishment = torch.clamp(at_lower_limit, min=0) * self.overextend_cost
        
        overextend_punishment = torch.sum(clipped_lower_punishment + clipped_upper_punishment, dim=1) / clipped_lower_punishment.shape[1]
        
        reward -= overextend_punishment
        
        
        # ---------- reward for the heading direction ----------
        
        target_direction_reward = batch_dot_product(self.target_direction, two_d_heading_direction) * self.target_direction_factor
        
        reward += target_direction_reward
    
        
        
        # -------------- cost of power --------------
        
        energy_punishment = torch.sum(self.actions ** 2, dim=-1) * self.energy_cost
        reward -= energy_punishment
        
        # ------------- cost of usign arms ------------
         
        arm_use_punishment = torch.sum(self.actions[:, self.upper_body_joint_indices]) / self.upper_body_joint_indices.shape[0] * self.arm_use_cost
        
        reward -= arm_use_punishment
        
        
        # ---------- has fallen or die on contact -------------
         
        terminations_height = self.death_height
        
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(self.root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
        
        summed_contact_forces = torch.sum(self.contact_forces, dim= 2) # sums x y and z components of contact forces together
        
        summed_contact_forces[:,self.left_foot_index] = 0.0
        summed_contact_forces[:, self.right_foot_index] = 0.0
        
        total_summed_contact_forces = torch.sum(summed_contact_forces, dim=1) # sum all the contact forces of the other indices together, to see if there is any other contact other than the feet
        
        has_contact = torch.where(total_summed_contact_forces > torch.zeros_like(total_summed_contact_forces), torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        if self.die_on_contact:
            has_fallen += has_contact
        else:
            n_times_contact = (summed_contact_forces > 0 ).to(dtype=torch.float32).sum(dim=1)
            
            contact_punishment = n_times_contact * self.contact_punishment_factor
            
            reward -= contact_punishment
        
        # ------------- cost for dying ----------
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(has_fallen == 1, - 1 * torch.ones_like(reward) * self.death_cost, reward)
        
    
        
        # average rewards per step
         
        upright_punishment = float(torch.mean(upright_punishment).item())
        target_velocity_reward = float(torch.mean(target_velocity_reward).item())
        jitter_punishment = - float(torch.mean(jitter_punishment).item())
        energy_punishment = - float(torch.mean(energy_punishment).item())
        arm_use_punishment = - float(torch.mean(arm_use_punishment).item())
        target_direction_reward = float(torch.mean(target_direction_reward).item())
        overextend_punishment = - float(torch.mean(overextend_punishment).item())
        if not self.die_on_contact:
            contact_punishment = -float(torch.mean(contact_punishment).item())
        else:
            contact_punishment = 0.0
        
        total_avg_reward = self.alive_reward + upright_punishment + target_velocity_reward + jitter_punishment + energy_punishment
        
        reward_constituents = {
                                'alive_reward': self.alive_reward,
                                'upright_punishment':  upright_punishment,
                                'target_velocity_reward':    target_velocity_reward,
                                'jitter_punishment':   jitter_punishment,
                                'target_direction_reward': target_direction_reward, 
                                'energy_punishment':   energy_punishment,
                                'arm_use_punishment': arm_use_punishment,
                                'overextend_punishment': overextend_punishment,
                                'contact_punishment': contact_punishment,
                                'total_reward': total_avg_reward
                            }
        
        
        return (reward, has_fallen, reward_constituents)
            
    
    
    def _add_to_env(self, env_ptr, env_id: int, robot_handle): 
        """During the _create_envs this is called to give mk1_envs the ability to add additional things to the environment

        Args:
            env_ptr (_type_): pointer to the env
        """
        
        camera_props = gymapi.CameraProperties()
        camera_props.width = 128
        camera_props.height = 128
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0,0.4,1.6)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
         
        self.gym.attach_camera_to_body(camera_handle, env_ptr,  robot_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        
        pass
    
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_mk1_terrain.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            
            
    def get_obs(self) -> Tuple[ Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return all the observations made by the actor and the critic

        Returns:
            Tuple[str, torch.Tensor]: _description_
        """
        
        # use jit script to compute the observations
        self.actor_obs["linear"][:], self.critic_obs["linear"][:] = compute_linear_robot_observations(
            root_states = self.root_states, 
            sensor_states=self.vec_force_sensor_tensor,
            dof_vel=self.dof_vel,
            dof_pos= self.dof_pos,
            dof_limits_lower=self.dof_limits_lower,
            dof_limits_upper=self.dof_limits_upper,
            dof_force= self.dof_force_tensor, 
            actions= self.actions
        )  
        
        self.actor_obs["command"][:, 0:2] = self.target_direction
        self.actor_obs["command"][:, 2] = self.target_velocity
        
    
    
    def get_num_playable_actors_per_env(self) -> int:
        """Return the amount of actors each environment has, this only includes actors, that are playable
        This distincion is stupid and only exists, because isaacgym currently does anot support any way of adding objects to environments, that are not actors

        Returns:
            int
        """
        return self.get_num_actors_per_env()
    
    
    def get_num_actors_per_env(self) -> int:
        """Get the total amount of actor per environment this includes non active actors like boxes or other inaminate matter

        Returns:
            int
        """
        
        return 1
    
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_actor_obs = 142
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, )),
            "command": gym.spaces.Box(low= -3.0, high= 5.0, shape= (3, ))
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

        num_critic_obs = 6
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_critic_obs, ))
        }) 
        
         
def compute_velocity_reward_factor(abs: torch.Tensor, zero_x_cord: Union[float, torch.Tensor], factor: Union[float, torch.Tensor]) -> torch.Tensor:
    """compute a function for the velocity reward factor
    https://www.geogebra.org/graphing/f2qxcw85

    Args:
        abs (torch.Tensor): the absolute tensor (x is only positive)
        zero_x_cord (torch.Tensor): the coordinate where the reward is zero
        factor (Union[float, torch.Tensor]): the y multiplication. This values is achieved for abs == 0
        
        f(x)=(((1)/(x ((1)/(2 zerop))+0.5))-1) * factor
    Returns:
        torch.Tensor: y
    """
    return  (1.0 / (abs * (1.0 / (2.0 * zero_x_cord))+ 0.5) - 1.0) * factor
     
    
 
@torch.jit.script
def compute_linear_robot_observations(root_states: torch.Tensor, 
                                sensor_states: torch.Tensor, 
                                dof_vel: torch.Tensor, 
                                dof_pos: torch.Tensor, 
                                dof_limits_lower: torch.Tensor,
                                dof_limits_upper: torch.Tensor,
                                dof_force: torch.Tensor,
                                actions: torch.Tensor
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
     
    
    linear_actor_obs = torch.cat((sensor_states.view(root_states.shape[0], -1), dof_pos, dof_vel, dof_limits_upper.tile(( root_states.shape[0], 1)),dof_limits_lower.tile(( root_states.shape[0],1 )),  dof_force, ang_velocity, torso_rotation, actions, torso_position), dim=-1)
    
    linear_critic_obs = torch.cat(( velocity, torso_position), dim=-1)
    
    return  linear_actor_obs,   linear_critic_obs




# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# terrain generator
from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, cfg, num_robots) -> None:

        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

