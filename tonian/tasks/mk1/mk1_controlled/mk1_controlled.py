from pydoc import cli
from gym.spaces import space
import numpy as np
from tonian.common.utils import join_configs 
from tonian.tasks.mk1.mk1_base import Mk1BaseClass
from tonian.tasks.base.vec_task import VecTask 
from tonian.common.torch_jit_utils import batch_dot_product

from tonian.common.spaces import MultiSpace


from typing import Dict, Any, Tuple, Union, Optional, List, Set

from isaacgym import gymtorch, gymapi

import yaml, os, torch, gym


class Mk1ControlledTask(Mk1BaseClass):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:
        
        base_config = self._get_standard_config()
        
        config = join_configs(base_config, config)
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
        
    def allocate_buffers(self):
        """Allocate tensors, that will be used throughout the simulation
        """
        super().allocate_buffers()
        self.update_reward_factor_buffers()
        
    def update_reward_factor_buffers(self):
        """
            Set the state names, the state probs and all the reward factor buffers
        """
        
        reward_weight_list = self.config["mk1_controlled"]["reward_weighting"]  
        self.state_names = [reward_block['name'] for reward_block in reward_weight_list]
        self.state_probabilities = [reward_block['prob'] for reward_block in reward_weight_list]
        
        self.command_state_distribution = torch.distributions.OneHotCategorical(torch.tensor(self.state_probabilities, dtype= torch.float32, device= self.device))
        self.command_state_tensor = self.command_state_distribution.sample(sample_shape=(self.num_envs, )).to(self.device).to(torch.int8)
        
        self.reward_factor_dict = dict([(state_block['name'], state_block['rewards'] ) for state_block in reward_weight_list])
        self.state_reward_factors = [state_block['rewards'] for state_block in reward_weight_list]
        
        self.all_reward_keys : Set = Mk1ControlledTask.get_all_reward_weighting_keys(reward_weight_list)
        
        # Set all the reward weighting buffers 
        for key in self.all_reward_keys:
            setattr(self, key, self.reward_weight_state_dependend_tensor(key))

        
    def reward_weight_state_dependend_tensor(self, key: str, dtype: torch.dtype = torch.float16):
        """Get the reward weights for a state dependent tensor

        Args:
            key (str): key 
            dtype (torch.dtype, optional): type of the tensor. Defaults to torch.float16.

        Returns:
            _type_: _description_
        """
        value_tensor = torch.zeros((self.num_envs,), dtype=dtype, device= self.device )
        for i in range(len(self.state_names)):
            value_tensor += self.command_state_tensor[:, i] * self.state_reward_factors[i].get(key, 0) 
        return value_tensor
    
    
    
    def update_reward_weight_state_dependend_tensor(self, key: str,  env_ids: torch.Tensor,  dtype: torch.dtype = torch.float16):
        """Get the reward weights for a state dependent tensor

        Args:
            key (str): key 
            dtype (torch.dtype, optional): type of the tensor. Defaults to torch.float16.

        Returns:
            _type_: _description_
        """
        value_tensor = torch.zeros((env_ids.shape[0],), dtype=dtype, device= self.device )
        for i in range(len(self.state_names)):
            value_tensor += self.command_state_tensor[env_ids, i] * self.state_reward_factors[i].get(key, 0) 
        return value_tensor
                
    
    
    def get_all_reward_weighting_keys(reward_weight_list: List[Dict]) -> Set:
        """Retuns the set of all the reward weighting keys

        Args:
            reward_weight_list (List[Dict]): list of the different configs

        Returns:
            Set: containing all keys 
        """
        result_set = set()
        
        for state_block in reward_weight_list:
            
            for key in state_block['rewards'].keys():
                result_set.add(key)
        return result_set
                 
    
        
        
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
         
        assert self.config["mk1_controlled"] is not None, "The mk1_controlled config must be set on the task config file"
        
        # reward_weight_dict = self.config["mk1_controlled"]["reward_weighting"]  
        
        # self.energy_cost = reward_weight_dict["energy_cost"]
        # self.directional_factor = reward_weight_dict["directional_factor"]
        # self.death_cost = reward_weight_dict["death_cost"]
        # self.alive_reward = float(reward_weight_dict["alive_reward"])
        # self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        # self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        # self.death_height = reward_weight_dict["death_height"]
        # self.overextend_cost = reward_weight_dict["overextend_cost"]
        # self.die_on_contact = reward_weight_dict.get("die_on_contact", True)
        # self.contact_punishment_factor = reward_weight_dict["contact_punishment"]
        # self.arm_position_cost = reward_weight_dict["arm_position_cost"]
        # 
        # self.arm_use_cost = reward_weight_dict['arm_use_cost']
        
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        super().reset_envs(env_ids)
        
        self.command_state_tensor[env_ids] = self.command_state_distribution.sample(sample_shape=(len(env_ids), )).to(self.device).to(torch.int8)
         
        # Set all the reward weighting buffers 
        for key in self.all_reward_keys:
            getattr(self, key)[env_ids] = self.update_reward_weight_state_dependend_tensor(key, env_ids)
            
        

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
        
            
        #heading_to_velocity_angle = torch.arccos( torch.dot(two_d_heading_direction, linear_velocity_x_y)  / vel_norm )
        heading_to_velocity_angle = torch.arccos( batch_dot_product(two_d_heading_direction, linear_velocity_x_y) / vel_norm)
        
        direction_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_norm * self.directional_factor, torch.zeros_like(reward))
    
        reward += direction_reward  
        
        # -------------- Punish for jittery motion (see ./research/2022-03-27_reduction-of-jittery-motion-in-action.md)--------------
        
        jitter_punishment = - torch.abs(self.actions - self.former_actions).view(reward.shape[0], -1).sum(-1) * self.jitter_cost
        reward += jitter_punishment
        
        
        
        #-------------- cost for overextension --------------
        distance_to_upper = self.dof_limits_upper - self.dof_pos
        distance_to_lower = self.dof_pos - self.dof_limits_lower
        distance_to_limit = torch.minimum(distance_to_upper, distance_to_lower)
        
        # 0.001 rad -> 0,071 deg 
        at_upper_limit = torch.where(distance_to_upper < 0.02, self.actions, torch.zeros_like(distance_to_limit))
        at_lower_limit = torch.where(distance_to_lower < 0.02, self.actions, torch.zeros_like(distance_to_lower)) * -1
        at_lower_limit[:, 8] = 0
        at_upper_limit[: , 8] = 0
        
        clipped_upper_punishment = torch.sum(torch.clamp(at_upper_limit, min=0), dim= 1) * self.overextend_cost
        clipped_lower_punishment = torch.sum(torch.clamp(at_lower_limit, min=0), dim = 1) * self.overextend_cost
        
        overextend_punishment = (clipped_lower_punishment + clipped_upper_punishment) * -1
        
        reward += overextend_punishment
        
       # ------------- cost of usign arms ------------
         
        arm_use_punishment = torch.abs(torch.sum(self.actions[:, self.upper_body_joint_indices], dim = 1) / self.upper_body_joint_indices.shape[0]) * self.arm_use_cost
        arm_use_punishment = arm_use_punishment * -1
        
        reward += arm_use_punishment
         
        arm_position_punishment = torch.square(torch.abs(torch.sum(self.dof_pos, dim=1))/ 1.51) * self.arm_position_cost 
        arm_position_punishment = arm_position_punishment * -1
        
        
        reward += arm_position_punishment
        
        
        
        # -------------- cost of power --------------
        
        energy_punishment = torch.sum(self.actions ** 2, dim=-1) * self.energy_cost
        energy_punishment = energy_punishment * -1
        
        reward += energy_punishment
         
        terminations_height = self.death_height
        
        
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(self.root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
        
        
        
        
        summed_contact_forces = torch.sum(self.contact_forces, dim= 2) # sums x y and z components of contact forces together
        
        summed_contact_forces[:,self.left_foot_index] = 0.0
        summed_contact_forces[:, self.right_foot_index] = 0.0
        
        total_summed_contact_forces = torch.sum(summed_contact_forces, dim=1) # sum all the contact forces of the other indices together, to see if there is any other contact other than the feet
        
        has_contact = torch.where(total_summed_contact_forces > torch.zeros_like(total_summed_contact_forces), torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        # if self.die_on_contact:
        #     has_fallen += has_contact
        # else:
        #     n_times_contact = (summed_contact_forces > 0 ).to(dtype=torch.float32).sum(dim=1)
        #     
        #     contact_punishment = n_times_contact * self.contact_punishment_factor
        #     
        #     reward -= contact_punishment
            
        
        # ------------- cost for dying ----------
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(has_fallen == 1, - 1 * torch.ones_like(reward) * self.death_cost, reward)
    
        
        # average rewards per step 
        reward_constituents = {**self.get_tensor_state_means("alive_reward", self.alive_reward),
                               **self.get_tensor_state_means("upright_punishment", upright_punishment),
                               **self.get_tensor_state_means("direction_reward", direction_reward),
                               **self.get_tensor_state_means("jitter_punishment", jitter_punishment), 
                               **self.get_tensor_state_means("overextend_punishment", overextend_punishment), 
                               **self.get_tensor_state_means("energy_punishment", energy_punishment), 
                               **self.get_tensor_state_means("total_reward", reward)}
        
        
        return (reward, has_fallen, reward_constituents)
        
         
         
    
    def get_tensor_state_means(self,input_name: str,  input: torch.Tensor) -> Dict[str, float]:
        """Calculate the mean of a tensor within a given state 

        Args:
            input (torch.Tensor): state based reward tensor
        """
        return {self.state_names[i] +'/'  +  input_name : torch.mean(input[self.command_state_tensor[:, i].to(torch.bool)]).item() for i in range(len(self.state_names)) }
        
        
    
    def _add_to_env(self, env_ptr, env_id: int, robot_handle): 
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
        base_config_path = os.path.join(dirname, 'config_mk1_controlled.yaml')
        
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
        
        self.actor_obs["command"] = self.command_state_tensor
    
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
            "command": gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ))
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



