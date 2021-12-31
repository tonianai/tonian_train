import torch
from torch.distributions.multivariate_normal import MultivariateNormal 
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
                 activation_fn: Type[nn.Module] = nn.ELU,
                 device: str = 'cuda:0'
                 ) -> None:
        """Create an instance of the Actor critic policy base class

        Args:
            actor_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the actor net has to take in, including commands
            critic_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the critic net has to take in, including commands
            action_size (int): The size of the continous one dimensional action vector
            action_std_init (float): Standatrd deviation of the multidim gaussian, from whch the action will be sampled
            activation_fn (Type[nn.Module], optional): The activation function used as standard throughout the network. Defaults to nn.ELU.
            device: Defaults to 'cuda:0'
        """
        super().__init__()
        
        self.actor_obs_shapes = actor_obs_shapes
        self.critic_obs_shapes = critic_obs_shapes
        self.activation_fn = activation_fn
        self.device = device
        self.action_size = action_size
        self.set_action_std(action_std_init)
        
    
    def set_action_std(self, new_action_std: float) -> None:
        """Creates an action variance for a given standard deviation.
        This Variance is used to sample from a multidimensinal gaussian as the output for an action
        Args:
            new_action_std (float): new standard deviation of the action
        """
        self.action_var = torch.full((self.action_size, ), new_action_std * new_action_std).to(self.device)
    
    def forward():
        raise NotImplementedError()

class SimpleActorCriticPolicy(ActorCriticPolicy):
    """The Simple Actor critic policy is an assymetric actor critic policy without an cnn and without rnn.
        The actor obs shapes are expected to be one dimensional and will be inserted in the same starting layer

    """
    
    def __init__(self, 
                 actor_obs_shapes: Tuple[Tuple[int, ...]], 
                 critic_obs_shapes: Tuple[Tuple[int, ...]], 
                 action_size: int, 
                 action_std_init: float, 
                 actor_hidden_layer_sizes: Tuple[int],
                 critic_hidden_layer_sizes: Tuple[int],
                 activation_fn: Type[nn.Module] = nn.ELU) -> None:
        """Create an instance of the Actor critic policy base class

        Args:
            actor_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the actor net has to take in, including commands
            critic_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the critic net has to take in, including commands
            action_size (int): The size of the continous one dimensional action vector
            action_std_init (float): Standatrd deviation of the multidim gaussian, from whch the action will be sampled
            activation_fn (Type[nn.Module], optional): The activation function used as standard throughout the network. Defaults to nn.ELU.
            actor_hidden_layer_sizes (Tuple[int]): The sizes of the layers, between the obs layer and the action vector
            critic_hidden_layer_sizes (Tuple[int]): The size of the layers, between the critic obs layer and the 1 dim value
            activation_fn (Type[nn.Module], optional): [description]. Defaults to nn.ELU.
        """
        super().__init__(actor_obs_shapes, critic_obs_shapes, action_size ,action_std_init, activation_fn=activation_fn)

        
        combined_actor_input_size = 0
        for input_space_shape in self.actor_obs_shapes:
            assert len(input_space_shape) == 1 , "The actor input spaces must all be one dimensional"
            combined_actor_input_size += input_space_shape[0]
            
        combined_critic_input_size = 0
        for input_space_shape in self.critic_obs_shapes:
            assert len(input_space_shape) == 1, "The critic input space must all be one dimensional"
            combined_critic_input_size += input_space_shape[0]
        
        assert combined_actor_input_size > 0 and combined_critic_input_size > 0 , "The input space cannot be 0 in size"
        
        # create the actor network
        layers_actor = []
        if len(actor_hidden_layer_sizes) == 0:
            # Simple One Layer Network without hidden layers
            layers_actor.append(nn.Linear(combined_actor_input_size, action_size))
        else:
            layers_actor.append(nn.Linear(combined_actor_input_size, actor_hidden_layer_sizes[0]))
            
            for i, size in enumerate(actor_hidden_layer_sizes):
                
                if i == 0:
                    continue
                
                # add the activation function to the layer
                layers_actor.append(self.activation_fn())
                
                # add the linear layer
                layers_actor.append(nn.Linear(actor_hidden_layer_sizes[-1], size))
   
            layers_actor.append(self.activation_fn())
            
            # The last layer must take action size into account
            layers_actor.append(nn.Linear(actor_hidden_layer_sizes[-1], action_size))
        
        # Add a last tanh function to map to output space between -1 and 1    
        layers_actor.append(nn.Tanh())
        
        # the asterix is unpacking all layers_actor items and passing them into the nn.Sequential
        self.actor = nn.Sequential(*layers_actor)
        
        # create the critic network
        layers_critic = []
        if len(critic_hidden_layer_sizes) == 0:
            # Simple One Layer Network without hidden layers -> maps to one value of the state
            layers_critic.append(nn.Linear(combined_critic_input_size, 1))
        else:
            layers_critic.append(nn.Linear(combined_critic_input_size, actor_hidden_layer_sizes[0]))
            
            for i, size in enumerate(actor_hidden_layer_sizes):
                
                if i == 0:
                    continue
                
                # add the activation function to the layer
                layers_critic.append(self.activation_fn())
                
                # add the linear layer
                layers_critic.append(nn.Linear(actor_hidden_layer_sizes[-1], size))
   
            layers_critic.append(self.activation_fn())
             
            layers_critic.append(nn.Linear(actor_hidden_layer_sizes[-1], 1))
            
        # the critic does not need a last layer
        
        # the asterix is unpacking all layers_critic items and passing them into the nn.Sequential
        self.actor = nn.Sequential(*layers_critic)
        
    def predict_action(self, actor_obs: Dict[str, torch.Tensor]):
        """Create an action using the actor network and a gaussian distribution
        -> the tensors will become detached after execution (only use for actiung eith the env)
        print(concat_obs.shape)
        Args:
            actor_obs (Dict[torch.Tensor]): Multispace output
        """
        
        concat_obs = None
        
        for key in actor_obs:
            
            if concat_obs:
                torch.cat((concat_obs, actor_obs[key]), dim=1)
            else:
                concat_obs = actor_obs[key]
                
                
        action_mean = self.actor(concat_obs)
        
        # the vairables are independend of each other within the distribution -> therefore diagonal matrix
        cov_matrix = torch.diag(self.action_var).unsqueeze(dim = 0)
        dist = MultivariateNormal(action_mean, cov_matrix)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach()
        
    def evaluate_action(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor], action: torch.Tensor):
        
        concat_obs = None
        
        for key in actor_obs:
            
            if concat_obs:
                torch.cat((concat_obs, actor_obs[key]), dim=1)
            else:
                concat_obs = actor_obs[key]
        
        action_mean = self.actor(actor_obs)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(critic_obs)

        return action_logprobs, state_values, dist_entropy 
    
    def forward(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor]):
        """Forward pass in all networks (actor and critic)

        Args:
            actor_obs (Dict[str, torch.Tensor]): [description]
            critic_obs (Dict[str): [description]
            
        Returns:
            action, value and log probability of that action
        """
        
        actor_concat_obs = None
        
        for key in actor_obs:
            if actor_concat_obs:
                torch.cat((actor_concat_obs, actor_obs[key]), dim=1)
            else:
                actor_concat_obs = actor_obs[key]
        
        critic_concat_obs = None
        
        for key in critic_obs:
            
            if critic_concat_obs:
                torch.cat((critic_obs, critic_obs[key]), dim= 1)
            else:
                critic_concat_obs = critic_obs[key]
                
        values = self.critic(critic_concat_obs)
        
        action_mean = self.actor(actor_concat_obs)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        actions = dist.sample()
        
        log_prob = dist.log_prob(actions)
                        
        return actions, values, log_prob
        
        