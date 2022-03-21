from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Tuple
import numpy as np
from tonian.common.spaces import MultiSpace

import gym, torch

class BaseAgents(ABC):
    """
    The base class for agents. (i.e Mk1)
    Instances do not include reward calculation, because this is subject to environment change
    BaseAgents class does include actor (and critic) observation calculation, becuase this is only subject to the agent.
    """
    
    def __init__(self, num_agents: int):
        pass
    
    
    @abstractmethod
    def create(self, num_agents: int, spacing: int, num_per_row: Optional[int]):
        """Create the agetns in the environment

        Args:
            num_agents (int): The number of tot
            spacing (int): _description_
            num_per_row (Optional[int]): _description_
        """
    
    @abstractmethod
    def reset(self, env_ids: np.ndarray):
        """
        Reset the agents of the given env_ids.
        Potentially add randomisation
        """
        
    @abstractmethod
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
    
    @abstractmethod
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
    
    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def act(self, action: torch.Tensor) -> None:
        """
        Apply the action to all the actors
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_observation(self) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get the observation of all actors

        Returns:
            Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]]]: (actor_obs, critic_obs)
        """
        
    
    
    
    
    
    
    