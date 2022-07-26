
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Tuple, Any, Optional
from tonian_train.common.spaces import MultiSpace
import gym, torch 
from gym import spaces


class VecTask(gym.Env, ABC):
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # The sum of all action dimensions        
        self._action_size = None
        
        self.num_envs = config["vec_task"]["num_envs"] 
        self.max_episode_length = self.config['vec_task'].get('max_episode_length', 10000)
 
    
    @abstractproperty
    def observation_space(self) -> gym.Space:
        return self.actor_observation_spaces.spaces
    
    @abstractproperty
    def action_space(self) -> gym.Space:
        pass

    @abstractproperty
    def reward_range(self):
        pass
        
        
        
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any], Optional[Dict[str, float]]]:
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]], Optional[Dict[str, float]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info, reward_constituents
        """
        
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the complete environment and return a output multispace
        Returns:
            Dict[str, torch.Tensor]: Output multispace (names in the dict correspond to those given in the multispace),
        """
        pass



    
    @abstractmethod
    def _get_observation_spaces(self) -> MultiSpace:
        """Define the different inputs the actor of the agent
         (this includes linear observations, viusal observations)
        
        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()
    
    
    @abstractmethod
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment properly
        """
        
        
    def set_tensorboard_logger(self, logger):
        self.logger = logger
        
        
    @property
    def action_size(self):
        """The sum over all the action dimension 
        """
        if not self._action_size:
             self._action_size =  self.action_space.sample().reshape(-1).shape[0]
        
        return self._action_size
             
        
        