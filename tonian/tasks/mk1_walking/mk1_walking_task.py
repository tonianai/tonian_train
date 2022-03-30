from gym.spaces import space
import numpy as np
from tonian.tasks.base.command import Command
from tonian.tasks.base.mk1.mk1_base import Mk1BaseClass
from tonian.tasks.base.vec_task import VecTask 
from tonian.common.utils.torch_jit_utils import batch_dot_product

from tonian.common.spaces import MultiSpace

from gym import spaces
import gym

from typing import Dict, Any, Tuple, Union, Optional

from isaacgym import gymtorch, gymapi

import yaml, os, torch, math


class Mk1WalkingTask(Mk1BaseClass):
    
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
        
        self.energy_cost = reward_weight_dict["energy_cost"]
        self.directional_factor = reward_weight_dict["directional_factor"]
        self.death_cost = reward_weight_dict["death_cost"]
        self.alive_reward = reward_weight_dict["alive_reward"]
        self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        
    
    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor,]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
               Tuple[torch.Tensor, torch.Tensor]: 
                   reward: torch.Tensor shape(num_envs, )
                   has_fallen: torch.Tensor shape(num_envs, )
                   constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
        """
        return   compute_robot_rewards(
            root_states= self.root_states,
            former_root_states= self.former_root_states, 
            actions= self.actions,
            former_actions= self.former_actions, 
            force_sensor_states=self.vec_force_sensor_tensor,
            alive_reward= self.alive_reward,
            death_cost= self.death_cost,
            directional_factor= self.directional_factor,
            energy_cost= self.energy_cost,
            upright_punishment_factor= self.upright_punishment_factor,
            jitter_cost= self.jitter_cost
        )
    
    
    def _add_to_env(self, env_ptr): 
        """During the _create_envs this is called to give mk1_envs the ability to add additional things to the environment

        Args:
            env_ptr (_type_): pointer to the env
        """
        pass
    
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

 
    

@torch.jit.script
def compute_robot_rewards(root_states: torch.Tensor,
                          former_root_states: torch.Tensor,
                          actions: torch.Tensor, 
                          former_actions: torch.Tensor,  
                          force_sensor_states: torch.Tensor,
                          death_cost: float,
                          alive_reward: float, 
                          directional_factor: float,
                          energy_cost: float,
                          upright_punishment_factor: float,
                          jitter_cost: float
                          )-> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute the reward and the is_terminals for the robot step in the environment 

    Args:
        root_states (torch.Tensor):   State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        former_root_states (torch.Tensor): State for each actor root from the last step: contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        actions (torch.Tensor): Actions taken: Shape(num_envs, ) + action_shape
        sensor_states (torch.Tensor): The states of the force sensors. Shape(num_envs, num_sensors, 6). or each sensor, the first three floats are the force and the last three floats are the torque. XYZ
        death_cost (float): _description_
        alive_reward (float): _description_
        directional_factor (float): _description_
        energy_cost (float): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            reward: torch.Tensor shape(num_envs, )
            has_fallen: torch.Tensor shape(num_envs, )
            constituents: Dict[str, number] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
    """
    
    
 
    
    # -------------- base reward for being alive --------------  
    
    reward = torch.ones_like(root_states[:, 0]) * alive_reward  
    
     
    quat_rotation = root_states[: , 3:7]
    
    #  -------------- reward for an upright torso -------------- 
    
    # The upright value ranges from 0 to 1, where 0 is completely horizontal and 1 is completely upright
    # Calulation explanation: 
    # take the first and the last value of the quaternion and take the quclidean distance
    upright_value = torch.sqrt(torch.sum( torch.square(quat_rotation[:, 0:4:3]), dim= 1 ))
    
    upright_punishment = (upright_value -1) * upright_punishment_factor
    
    reward += upright_punishment
    
    #  -------------- reward for speed in the right heading direction -------------- 
    
    linear_velocity_x_y = root_states[: , 7:9]
    
    # direction_in_deg base is -> neg x Axis
    direction_in_deg_to_x = torch.acos(quat_rotation[:, 0]) * 2
    
    # unit vecor of heading when seen from above 
    # this unit vector makes little sense, when the robot is highly non vertical
    two_d_heading_direction = torch.transpose(torch.cat((torch.unsqueeze(torch.sin(direction_in_deg_to_x), dim=0), torch.unsqueeze(torch.cos(direction_in_deg_to_x),dim=0) ), dim = 0), 0, 1)
    
    # compare the two_d_heading_direction with the linear_velocity_x_y using the angle between them
    # magnitude of the velocity (2 norm)
    vel_norm = torch.linalg.vector_norm(linear_velocity_x_y, dim=1)
     
        
    #heading_to_velocity_angle = torch.arccos( torch.dot(two_d_heading_direction, linear_velocity_x_y)  / vel_norm )
    heading_to_velocity_angle = torch.arccos( batch_dot_product(two_d_heading_direction, linear_velocity_x_y) / vel_norm)
     
    direction_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_norm * directional_factor, torch.zeros_like(reward))
   
    reward += direction_reward  
    
    # -------------- Punish for jittery motion (see ./research/2022-03-27_reduction-of-jittery-motion-in-action.md)--------------
    
    jitter_punishment = torch.abs(actions - former_actions).view(reward.shape[0], -1).sum(-1) * jitter_cost
    reward -= jitter_punishment
      
     
    # -------------- cost of power --------------
    
    energy_punishment = torch.sum(actions ** 2, dim=-1) * energy_cost
    reward -= energy_punishment
     
    # -------------- punish for having fallen -------------- 
    terminations_height = 1.25
    # root_states[:, 2] defines the y positon of the root body 
    reward = torch.where(root_states[:, 2] < terminations_height, - 1 * torch.ones_like(reward) * death_cost, reward)
    
    #print(reward[10:])
    
    has_fallen = torch.zeros_like(reward, dtype=torch.int8)
    has_fallen = torch.where(root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
    
    
    # average rewards per step
    
    alive_reward = float(alive_reward)
    upright_punishment = float(torch.mean(upright_punishment).item())
    direction_reward = float(torch.mean(direction_reward).item())
    jitter_punishment = - float(torch.mean(jitter_punishment).item())
    energy_punishment = - float(torch.mean(energy_punishment).item())
    
    total_avg_reward = alive_reward + upright_punishment + direction_reward + jitter_punishment + energy_punishment
    
    reward_constituents = {
                            'alive_reward': alive_reward,
                            'upright_punishment':  upright_punishment,
                            'direction_reward':    direction_reward,
                            'jitter_punishment':   jitter_punishment,
                            'energy_punishment':   energy_punishment,
                            'total_reward': total_avg_reward
                           }
    
    return (reward, has_fallen, reward_constituents)


