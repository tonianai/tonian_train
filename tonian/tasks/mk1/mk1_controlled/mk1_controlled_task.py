from pydoc import cli
from gym.spaces import space
import numpy as np
from tonian.common.utils import join_configs
from tonian.tasks.common.command import Command
from tonian.tasks.mk1.mk1_base import Mk1BaseClass
from tonian.tasks.base.vec_task import VecTask 
from tonian.common.torch_jit_utils import batch_dot_product
from tonian.tasks.common.task_dists import task_dist_from_config

from tonian.common.spaces import MultiSpace


from typing import Dict, Any, Tuple, Union, Optional

from isaacgym import gymtorch, gymapi

import yaml, os, torch, math, gym


class Mk1ControlledTask(Mk1BaseClass):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:
        
        base_config = self._get_standard_config()
        
        config = join_configs(base_config, config)
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
        
        
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
         
        assert self.config["mk1_controlled"] is not None, "The mk1_walking config must be set on the task config file"
        
        reward_weight_dict = self.config["mk1_controlled"]["reward_weighting"]  
        
        self.energy_cost = reward_weight_dict["energy_cost"]
        self.directional_factor = reward_weight_dict["directional_factor"]
        self.death_cost = reward_weight_dict["death_cost"]
        self.alive_reward = float(reward_weight_dict["alive_reward"])
        self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        self.death_height = reward_weight_dict["death_height"]
        self.overextend_cost = reward_weight_dict["overextend_cost"]
        self.die_on_contact = reward_weight_dict.get("die_on_contact", True)
        self.contact_punishment_factor = reward_weight_dict["contact_punishment"]
        self.target_terminate_distance = reward_weight_dict['target_terminate_distance']
        self.target_reached_reward_factor = reward_weight_dict['target_reached_reward_factor']
        self.forward_factor = reward_weight_dict['forward_direction_factor']
        
        target_positions = self.config["mk1_controlled"]["target_positions"]
        self.target_x_dist = target_positions['x_pos']
        self.target_y_dist = target_positions['y_pos']
        
        
    def get_additional_critic_obs(self) -> Dict[str, torch.Tensor]:
        return {}
    
    def get_additional_actor_obs(self) -> Dict[str, torch.Tensor]:
        # commands 0 -> determines whether the actor should be idle
        # command 1 & 2 -> position oth the target
        commands = torch.cat((self.is_idle.unsqueeze(dim = 1), self.target_pos), dim = 1)
        
        return {'command': commands}
        
    def _get_actor_observation_spaces(self) -> MultiSpace:
        num_actor_obs = 108
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, )),
            "command": gym.spaces.Box(low=-10.0, high = 10.0, shape = (3, ))
        })
        
    def allocate_buffers(self):
        super().allocate_buffers()
    
        self.is_idle = torch.zeros((self.num_envs, ), device= self.device)
        self.x_target_mean = torch.ones((self.num_envs, ), device= self.device) * self.target_x_dist['mean']
        self.x_target_std = torch.ones((self.num_envs, ), device= self.device) * self.target_x_dist['std']
        x_cordinate = torch.normal(mean= self.x_target_mean, std= self.x_target_std).unsqueeze(dim=1)
        
               
        self.y_target_mean = torch.ones((self.num_envs, ), device= self.device) * self.target_y_dist['mean']
        self.y_target_std = torch.ones((self.num_envs, ), device= self.device) * self.target_y_dist['std']
        y_cordinate = torch.normal(mean= self.y_target_mean, std= self.y_target_std).unsqueeze(dim=1)
        
        self.target_pos =  torch.cat((x_cordinate, y_cordinate), dim= 1)
        
        
        
        
        
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        
        
        self.target_pos[env_ids, 0] = torch.normal(mean= self.x_target_mean, std= self.x_target_std)[env_ids]
        self.target_pos[env_ids, 1] = torch.normal(mean= self.y_target_mean, std= self.y_target_std)[env_ids]
        
        super().reset_envs(env_ids)
        
    

    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.target_asset = self.gym.create_sphere(self.sim, 0.5, asset_options)
        
        return super()._create_envs(spacing, num_per_row)
    
    def _get_amount_of_actors_per_env(self):
        return 2
    
        
      
    def _add_to_env(self, env_ptr, env_id: int ): 
        """During the _create_envs this is called to give mk1_envs the ability to add additional things to the environment

        Args:
            env_ptr (_type_): pointer to the env
            env_id (int): int of the env
        """
        print("add to env ptr")
        print(env_id)
        
        pose = gymapi.Transform()
        print(pose.p)
        
        #target_asset = self.gym.create_actor(env_ptr, self.target_asset, pose , "target", env_id, 1, 1)
        
        
        
        
        pass

    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor,]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
               Tuple[torch.Tensor, torch.Tensor]: 
                   reward: torch.Tensor shape(num_envs, )
                   has_fallen: torch.Tensor shape(num_envs, )
                   constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
        """
    
        do_terminate = torch.zeros_like(self.root_states[:, 0], dtype= torch.int8, device= self.device)
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
        
        direction_to_target = self.root_states[: , 7:9] - self.target_pos
        distance_to_target = torch.sqrt(torch.square(direction_to_target).sum(dim=1))
        direction_to_target = torch.nn.functional.normalize(direction_to_target)
        
        linear_velocity_x_y = self.root_states[: , 7:9]
        
        # direction_in_deg base is -> neg x Axis
        direction_in_deg_to_x = torch.acos(quat_rotation[:, 0]) * 2
        
        # unit vecor of heading when seen from above 
        # this unit vector makes little sense, when the robot is highly non vertical
        two_d_heading_direction = torch.transpose(torch.cat((torch.unsqueeze(torch.sin(direction_in_deg_to_x), dim=0), torch.unsqueeze(torch.cos(direction_in_deg_to_x),dim=0) ), dim = 0), 0, 1)
        
        # compare the two_d_heading_direction with the linear_velocity_x_y using the angle between them
        # magnitude of the velocity (2 norm)
        vel_norm = torch.linalg.vector_norm(linear_velocity_x_y, dim=1)
        
            
        #heading_to_velocity_angle = torch.arccos( torch.dot(two_d_heading_direction, linear_velocity_x_y)  / vel_norm )
        heading_to_velocity_angle = torch.arccos( batch_dot_product(linear_velocity_x_y, direction_to_target) / vel_norm)
        
        direction_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_norm * self.directional_factor, torch.zeros_like(reward))

        reward += direction_reward
    

        # -------------- reward for forward direction ----------
        
        heading_to_velocity_angle = torch.arccos( batch_dot_product(two_d_heading_direction, linear_velocity_x_y) / vel_norm)
        
        forward_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_norm * self.forward_factor, torch.zeros_like(reward))
    
    
        reward += forward_reward  
        
        # ------------- reward for reaching target --------------
        
        has_target_reached = torch.where(distance_to_target < self.target_terminate_distance, torch.ones_like(reward), torch.zeros_like(reward)) 
        target_reach_reward = has_target_reached* self.target_reached_reward_factor
        
        do_terminate += has_target_reached.to(dtype=torch.int8)
        reward += target_reach_reward
        
        
        
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
        
    
        
        
        # -------------- cost of power --------------
        
        energy_punishment = torch.sum(self.actions ** 2, dim=-1) * self.energy_cost
        reward -= energy_punishment
         
        terminations_height = self.death_height
        
        
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(self.root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
        
        do_terminate += has_fallen
        
        
        summed_contact_forces = torch.sum(self.contact_forces, dim= 2) # sums x y and z components of contact forces together
        
        summed_contact_forces[:,self.left_foot_index] = 0.0
        summed_contact_forces[:, self.right_foot_index] = 0.0
        
        total_summed_contact_forces = torch.sum(summed_contact_forces, dim=1) # sum all the contact forces of the other indices together, to see if there is any other contact other than the feet
        
        has_contact = torch.where(total_summed_contact_forces > torch.zeros_like(total_summed_contact_forces), torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        if self.die_on_contact:
            do_terminate += has_contact
        else:
            n_times_contact = (summed_contact_forces > 0 ).to(dtype=torch.float32).sum(dim=1)
            
            contact_punishment = n_times_contact * self.contact_punishment_factor
            
            reward -= contact_punishment
        
        # ------------- cost for dying ----------
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(has_fallen == 1, - 1 * torch.ones_like(reward) * self.death_cost, reward)
        
        
        
        # average rewards per step
         
        upright_punishment = float(torch.mean(upright_punishment).item())
        direction_reward = float(torch.mean(direction_reward).item())
        jitter_punishment = - float(torch.mean(jitter_punishment).item())
        energy_punishment = - float(torch.mean(energy_punishment).item())
        forward_reward = float(torch.mean(forward_reward).item())
        target_reach_reward = float(torch.mean(target_reach_reward).item())
        overextend_punishment = - float(torch.mean(overextend_punishment).item())
        if not self.die_on_contact:
            contact_punishment = -float(torch.mean(contact_punishment).item())
        else:
            contact_punishment = 0.0
        
        total_avg_reward = self.alive_reward + upright_punishment + direction_reward + jitter_punishment + energy_punishment
        
        reward_constituents = {
                                'alive_reward': self.alive_reward,
                                'upright_punishment':  upright_punishment,
                                'target_reach_reward': target_reach_reward, 
                                'direction_reward':    direction_reward,
                                'forward_reward': forward_reward,
                                'jitter_punishment':   jitter_punishment,
                                'energy_punishment':   energy_punishment,
                                'overextend_punishment': overextend_punishment,
                                'contact_punishment': contact_punishment,
                                'total_reward': total_avg_reward
                            }
        
        
        return (reward, do_terminate, reward_constituents)
            
  
    
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_mk1_controlled.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            

 