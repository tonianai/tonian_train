import torch 
import torch.nn as nn


from typing import Union, Dict, Tuple, Type, List

from gym import spaces
import gym
from elysium.common.utils.spaces import MultiSpace, SingleOrMultiSpace

from abc import ABC, abstractmethod

class ActorCriticPolicy(nn.Module, ABC):
    """The Actor Critic policy assumes a asymmetric actzor critic, because a symmetric actor critic is just a 

    Args:
        nn ([type]): [description]
    """
    
    
    def __init__(self,
                 actor_obs_shapes: Tuple[Tuple[int, ...]], 
                 critic_obs_shapes: Tuple[Tuple[int, ...]],
                 action_size: int,
                 action_std_init: float,
                 activation_fn: Type[nn.Module] = nn.ELU
                 ) -> None:
        """Create an instance of the Actor critic policy base class

        Args:
            actor_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the actor net has to take in, including commands
            critic_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the critic net has to take in, including commands
            action_size (int): The size of the continous one dimensional action vector
            action_std_init (float): Standatrd deviation of the multidim gaussian, from whch the action will be sampled
            activation_fn (Type[nn.Module], optional): The activation function used as standard throughout the network. Defaults to nn.ELU.
        """
        super().__init__()
        
        self.actor_obs_shapes = actor_obs_shapes
        self.critic_obs_shapes = critic_obs_shapes
        self.action_std_init = action_std_init
        self.activation_fn = activation_fn
        
        
    def forward():
        raise NotImplementedError()

class SimpleActorCriticPolicy(ActorCriticPolicy):
    """The Simple Actor critic policy is an assymetric actor critic policy without an cnn and without rnn

    """
    
    def __init__(self, 
                 actor_obs_shapes: Tuple[Tuple[int, ...]], 
                 critic_obs_shapes: Tuple[Tuple[int, ...]], 
                 action_size: int, 
                 action_std_init: float, 
                 actor_network_layer_sizes: Tuple[int],
                 critic_network_layer_sizes: Tuple[int],
                 activation_fn: Type[nn.Module] = nn.ELU) -> None:
        """Create an instance of the Actor critic policy base class

        Args:
            actor_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the actor net has to take in, including commands
            critic_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the critic net has to take in, including commands
            action_size (int): The size of the continous one dimensional action vector
            action_std_init (float): Standatrd deviation of the multidim gaussian, from whch the action will be sampled
            activation_fn (Type[nn.Module], optional): The activation function used as standard throughout the network. Defaults to nn.ELU.
            actor_network_layer_sizes (Tuple[int]): The sizes of the layers, between the obs layer and the action vector
            critic_network_layer_sizes (Tuple[int]): The size of the layers, between the critic obs layer and the 1 dim value
            activation_fn (Type[nn.Module], optional): [description]. Defaults to nn.ELU.
        """
        super().__init__(actor_obs_shapes, critic_obs_shapes, action_size ,action_std_init, activation_fn=activation_fn)
    
    
        # create the actor network
        
        
   
        
    
    