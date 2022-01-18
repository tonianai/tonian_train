""" Policy Base class and concrete implementations based on the implementation of stable baselines3"""
 
from typing import Dict, Union, Tuple, Any, List, Type, Optional

import torch
import torch.nn as nn

from functools import partial

import numpy as np

from abc import ABC, abstractmethod

from gym import spaces

import collections
import warnings

from elysium.common.distributions import DiagGaussianDistribution, Distribution
from elysium.common.spaces import MultiSpace
from elysium.common.utils.utils import Schedule
from elysium.common.networks import BaseNet
from elysium.common.type_aliases import Observation
class BasePolicy(nn.Module , ABC):
    
    def __init__(self, 
                 action_space: spaces.Space,
                 device: Union[torch.device, str],
                 squash_actions: bool = False,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Abstract Base class for all neural network policies
        Args:
            action_space (spaces.Space):
                the output space of the neural network for the actions
                Only spaces are supported at this time, multispaces for for example communication between robots could be added in the future
            device: the torch device
            sqash_actions: determines whether the action should be squashed with the tanh function
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer in use. Defaults to torch.optim.Adam.
            optimizer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the optimizer class. Defaults to None.
            
        """
        super().__init__()
         
        self.action_space = action_space
        self.squash_actions = squash_actions

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
    def _predict(self,actor_obs: Observation, determinstic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        
    def predict(self, 
                actor_obs: Observation,
                state: Optional[Tuple[torch.Tensor, ...]],
                deterministic: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        TODO: Implement hidden states
        Args:
            actor_obs (Observation):
            state (Optional[Tuple[torch.Tensor, ...]]): [For recurrent policies]
            deterministic (bool, optional): [description]. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]: [description]
        """
        self.set_training_mode(False)
        
        with torch.no_grad():
            actions = self._predict(actor_obs, determinstic= deterministic)
            
        if isinstance(self.action_space, spaces.Box):
            
            if self.squash_actions:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # actions could be on an arbitrary scale -> clip the actions to avoid out of bound error
                actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        return actions, state
        
    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)
        
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
    
    
    
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
                 squash_actions: bool = False,
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
        
        super().__init__(action_space,
                         device,
                         squash_actions=squash_actions,
                         optimizer_class=optimizer_class, 
                         optimizer_kwargs=optimizer_kwargs)
        
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
            "squash_actions": self.squash_actions,
            "optimizer_class": self.optimizer_class,
            "optimizer_kwargs": self.optimizer_kwargs
            
        }
    
    def forward(self,
                actor_obs: Observation,
                critic_obs: Optional[Observation],
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all networks(actor and critic)

        Args:
            actor_obs (Observation): Observation of the actor
            critic_obs (Optional[Observation]): Observations of the crtitic
            deterministic (bool, optional): Whether to sample or use deterministic actions. Defaults to False.
        
        Returns:
            action: torch.Tensor
            value: torch.Tensor
            log_prob: torch.Tensor The log probablility of the action
        
        """
        if critic_obs is None:
            critic_obs = actor_obs
        
        # pass the actor observations troguh the actor net
        latent_pi = self.actor_net(actor_obs)
         
        # pass the critic observations trough the critic net 
        latent_vf = self.critic_net(critic_obs)
        
        # Evaluate values
        values = self.value_latent_net(latent_vf)
         
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # sample from the distribution
        actions = distribution.get_actions(deterministic= deterministic)

        log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs
        
    def _predict(self, actor_obs: Observation, deterministic: bool = False):
        """
        
        Args: 
            actor_obs (Optional[Observation]): Observations of the actor
            deterministic (bool, optional): Whether to sample or use deterministic actions. Defaults to False.
        
        """
        # sample from the distribution
        return self.get_distribution(actor_obs).get_actions(deterministic= deterministic)
    

    
    
    def evaluate_actions(self, 
                         actor_obs: Observation,
                         critic_obs: Observation,
                         action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the actions according to the current policy

        Args:
            actor_obs (Observation)
            critic_obs (Observation)
            action (torch.Tensor)

        Returns:
            estimated_value (torch.Tensor)
            log_likelihood (torch.Tensor)
            entropy (torch.Tensor)
        """
        latent_pi = self.actor_net(actor_obs)
        latent_vf = self.critic_net(critic_obs)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(action)
        values = self.value_latent_net(latent_vf)
        return values, log_prob, distribution.entropy()
    
    def predict_values(self, critic_obs: Observation) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observation

        Args:
            critic_obs (Observation)

        Returns:
            torch.Tensor: the estimated value
        """
        latent_vf = self.critic_net(critic_obs)
        return self.value_latent_net(latent_vf)
        
        
    def get_distribution(self,actor_obs: Observation) -> Distribution:
        """
        Get the current action policy distribution given the observation

        Args:
            actor_obs (Observation): Observations meant for the actor
        """
        
        latent_pi = self.actor_net(actor_obs)
        # get the mean value for the action distribution
        return self._get_action_dist_from_latent(latent_pi)
        
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_latent_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            raise ValueError("Invalid action distribution")
        
    
    def build(self, init_lr: float):
        """Initialize the networks
            - shared_net
            - critic_net
            - actor_net
        """
        
        # get the implementations from child classes
        self.actor_net = self.build_actor_net()
        self.critic_net = self.build_critic_net()
        
        self.latent_dim_pi = self.actor_net.latent_dim
        self.latent_dim_vf = self.critic_net.latent_dim
        
        # TODO: add different distributions if needed
        
        # the action latent net connectes the action net with the action size
        # the log dist 
        self.action_latent_net, self.log_std = self.action_dist.proba_distribution_net(
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
        