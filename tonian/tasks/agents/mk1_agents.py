

from typing import Dict, Tuple, Union, Optional

import torch, os, gym

import numpy as np
 
from tonian.tasks.agents.base_agents import  BaseAgents
from tonian.common.spaces import MultiSpace

from isaacgym import gymapi, gymtorch

class Mk1Agents(BaseAgents):
    
    def __init__(self, num_agents: int, gym, sim):
        super().__init__(num_agents, gym, sim)
        
        
    def create(self, spacing: int, num_per_row: Optional[int]):
        """Create the agetns in the environment

        Args:
            num_agents (int): The number of tot
            spacing (int): _description_
            num_per_row (Optional[int]): _description_
        """
        
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf/mk-1/")
        mk1_robot_file = "robot.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        
        mk1_robot_asset = self.gym.load_asset(self.sim, asset_root, mk1_robot_file, asset_options)
        
        self.num_dof = self.gym.get_asset_dof_count(mk1_robot_asset)
        
    
    
    def reset(self, env_ids: np.ndarray):
        """
        Reset the agents of the given env_ids.
        Potentially add randomisation
        """
         
    def get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()
     
    def get_critic_observation_spaces(self) -> MultiSpace:
        """Define the different observations for the critic of the agent
        
        
         The observations will later be combined with other inputs like commands to create the critic input space
        
        This is an asymmetric actor critic implementation  -> The critic observations differ from the actor observations
        and unlike the actor inputs the actor inputs don't have to be things that a real life robot could also observe in inference.
        
        Things like distance to target position, that can not be observed on site can be included in the critic input
    
        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()
     
    def get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        raise NotImplementedError()
     
    def act(self, action: torch.Tensor) -> None:
        """
        Apply the action to all the actors
        """
        raise NotImplementedError()
     
    def get_observation(self) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get the observation of all actors

        Returns:
            Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]]]: (actor_obs, critic_obs)
        """
        
    