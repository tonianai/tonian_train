""" Policy Base class and concrete implementations based on the implementation of stable baselines3"""

from typing import Dict, Union, Tuple, Any, List, Type, Optional

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from gym import spaces

import collections
import warnings

from elysium.common.distributions import DiagGaussianDistribution
from elysium.common.spaces import MultiSpace
from elysium.common.utils.utils import Schedule

class BasePolicy(nn.Module , ABC):
    
    def __init__(self, 
                 action_space: spaces.Space,
                 device: Union[torch.device, str],
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Abstract Base class for all neural network policies
        Args:
            action_space (spaces.Space):
                the output space of the neural network for the actions
                Only spaces are supported at this time, multispaces for for example communication between robots could be added in the future
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer in use. Defaults to torch.optim.Adam.
            optimizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the optimizer class. Defaults to None.
            
        """
        super().__init__()
         
        self.action_space = action_space

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]
        
        self.device = device
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
class ActorCriticPolicy(BasePolicy, ABC):
    
    def __init__(self,
                 actor_obs_space: Union[spaces.Space, MultiSpace],
                 critic_obs_space: Optional[Union[spaces.Space, MultiSpace]],
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 init_log_std: float, 
                 device: Union[torch.device, str],
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """The actor critic policy has in addition to having an action output also a value output

        Args:
            actor_obs_space (Union[spaces.Space, MultiSpace]): The observation space used as an input for the actor
            critic_obs_space (Optional[Union[spaces.Space, MultiSpace]]): The observation space used as an input for the critic 
            -> if it is null the critic net uses the same observation space as the actor net and the actor critic is symmetric
            action_space (spaces.Space): 
                the output space of the neural network for the actions
                Only spaces are supported at this time, multispaces for for example communication between robots could be added in the future
            lr_schedule (Schedule): 
                Schedule for chaning the learning rate of the optimizer dynamically
            init_log_std (float): 
                Initial log of the standard deviation for the action distribution
            device (Union[torch.device, str]): device
            optimizer_class (Type[torch.optim.Optimizer], optional): optimizer class used. Defaults to torch.optim.Adam.
            optimizer_kwargs (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
        """
        
        super().__init__(action_space, device, optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs)
        
        self.actor_obs_space = actor_obs_space
        self.critic_obs_space = critic_obs_space
        
        self.lr_schedule = lr_schedule
        self.init_log_std = init_log_std
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        
        self.action_dist = DiagGaussianDistribution()
    
    def forward(self, actor_obs: torch.Tensor,  critic_obs: Optional[torch.Tensor]):
        pass
        
    
    

