""" Policy Base class and concrete implementations based on the implementation of stable baselines3"""

from lib2to3.pytree import Base
from typing import Dict, Union, Tuple, Any, List, Type, Optional

import torch
import torch.nn as nn

from functools import partial

import numpy as np

from abc import ABC, abstractmethod

from gym import spaces

import collections
import warnings

from elysium.common.distributions import DiagGaussianDistribution
from elysium.common.spaces import MultiSpace
from elysium.common.utils.utils import Schedule
from elysium.common.networks import BaseNet
from elysium.common.type_aliases import Observation
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
        self.optimizer = None  # type: Optional[torch.optim.Optimizer]
        
        self.device = device
        
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
    
    
    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)
        
    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "auto") -> "BasePolicy":
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        saved_variables = torch.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model
    
    @abstractmethod
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
    
class ActorCriticPolicy(BasePolicy, ABC):
    
    def __init__(self,
                 actor_obs_space: Union[spaces.Space, MultiSpace],
                 critic_obs_space: Optional[Union[spaces.Space, MultiSpace]],
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 init_log_std: float = 0, 
                 device: Union[torch.device, str] = "cuda:0",
                 ortho_init: bool = True,
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
        
        self.ortho_init = ortho_init
        
        self.lr_schedule = lr_schedule
        self.init_log_std = init_log_std
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        
        self.action_dist = DiagGaussianDistribution()
        self.build(lr_schedule(1))
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return {
            "actor_obs_space": self.actor_obs_space,
            "critic_obs_space": self.critic_obs_space,
            "action_space":self.action_space,
            "lr_schedule": self.lr_schedule,
            "init_log_std": self.init_log_std,
            "ortho_init": self.ortho_init,
            "device":self.device,
            "optimizer_class": self.optimizer_class,
            "optimizer_kwargs": self.optimizer_kwargs
            
        }
    
    def forward(self, actor_obs: Observation, critic_obs: Optional[Observation]):
        pass
        
    def predict(self, actor_obs: Observation):
        """
        
        Args:
            actor_obs (torch.Tensor): [description]
        """
        
        
    
    def build(self, init_lr: float):
        """Initialize the networks
            - shared_net
            - critic_net
            - actor_net
        """
        
        # get the implementations from child classes
        self.actor_net = self.build_actor_net()
        self.critic_net = self.build_critic_net()
        self.shared_net = self.build_shared_net()
        
        self.latent_dim_pi = self.actor_net.latent_dim
        self.latent_dim_vf = self.critic_net.latent_dim
        
        # TODO: add different distributions if needed
        
        # the action latent net connectes the action net with the action size
        # the log dist 
        self.action_latent_net, self.log_dist = self.action_dist.proba_distribution_net(
            latent_dim= self.latent_dim_pi,
            log_std_init= self.init_log_std
        )
        
        # value latent net combines the output of the value net to a single dim
        self.value_latent_net = nn.Linear(self.latent_dim_vf, 1)
        
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            self.shared_net.appy(partial(self.init_weights, gain = np.sqrt(2)))
            self.actor_net.appy(partial(self.init_weights, gain = np.sqrt(2)))
            self.critic_net.appy(partial(self.init_weights, gain = np.sqrt(2))) 
            
            self.action_latent_net.appply(partial(self.init_weights, gain=1.0))
            self.value_latent_net.appply(partial(self.init_weights, gain=0.01))
            
        self.optimizer = self.optimizer_class(self.parameters(), lr=init_lr, **self.optimizer_kwargs)
            
            
        
    
    @abstractmethod
    def build_actor_net(self) -> BaseNet:
        """Build the actor net in the implementation of the actor critic algorithm 
        
        Returns:
            BaseNet: actor_net
        """
        pass
    
    @abstractmethod
    def build_critic_net(self) -> BaseNet:
        """Build the critic in the implementaion of the actor critic algorithm

        Returns:
            BaseNet: [description]
        """
        
        pass
    
    @abstractmethod
    def build_shared_net(self) -> BaseNet:
        """Build the network, that is shared between the actor and the critic 
        
        Returns:
            BaseNet: [description]
        """
        
        pass
        