import torch
from torch.distributions import Normal 
import torch.nn as nn


from typing import Union, Dict, Tuple, Type, List, Optional, Any   

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
                 lr_init: float,
                 log_std_init: float = 0,
                 activation_fn: Type[nn.Module] = nn.ELU,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, 
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
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
        self.log_std_init = log_std_init
        self.lr_init = lr_init
        
        self.log_std = nn.Parameter(torch.ones(self.action_size) * self.log_std_init, requires_grad=True)
        
        #self.register_parameter("fdsjkdskfl", self.log_std)
        
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        
        
        
    def _build_optim(self):
        """Set the optimizer"""
        self.optimizer = self.optimizer_class(self.parameters(),lr=self.lr_init, **self.optimizer_kwargs)
        

    
    
    def forward():
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    

class SimpleActorCriticPolicy(ActorCriticPolicy):
    """The Simple Actor critic policy is an assymetric actor critic policy without an cnn and without rnn.
        The actor obs shapes are expected to be one dimensional and will be inserted in the same starting layer

    """
    
    def __init__(self, 
                 actor_obs_shapes: Tuple[Tuple[int, ...]], 
                 critic_obs_shapes: Tuple[Tuple[int, ...]], 
                 action_size: int, 
                 lr_init: float,
                 actor_hidden_layer_sizes: Tuple[int],
                 critic_hidden_layer_sizes: Tuple[int],
                 log_std_init: float = 0, 
                 activation_fn: Type[nn.Module] = nn.ELU,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: str = "cuda:0"
                 ) -> None:
        """Create an instance of the Actor critic policy base class

        Args:
            actor_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the actor net has to take in, including commands
            critic_obs_shapes (Tuple[Tuple[int, ...]]): The shapes the critic net has to take in, including commands
            action_size (int): The size of the continous one dimensional action vector
            std_init (float): Standatrd deviation of the multidim gaussian, from whch the action will be sampled
            activation_fn (Type[nn.Module], optional): The activation function used as standard throughout the network. Defaults to nn.ELU.
            actor_hidden_layer_sizes (Tuple[int]): The sizes of the layers, between the obs layer and the action vector
            critic_hidden_layer_sizes (Tuple[int]): The size of the layers, between the critic obs layer and the 1 dim value
            activation_fn (Type[nn.Module], optional): [description]. Defaults to nn.ELU.
        """
        super().__init__(actor_obs_shapes, 
                         critic_obs_shapes, 
                         action_size,
                         lr_init, 
                         log_std_init= log_std_init,
                         activation_fn=activation_fn, 
                         optimizer_class=optimizer_class,
                         optimizer_kwargs= optimizer_kwargs,
                         device=device)

        
        combined_actor_input_size = 0
        for input_space_shape in self.actor_obs_shapes:
            assert len(input_space_shape) == 1 , "The actor input spaces must all be one dimensional"
            combined_actor_input_size += input_space_shape[0]
            
        combined_critic_input_size = 0
        for input_space_shape in self.critic_obs_shapes:
            assert len(input_space_shape) == 1, "The critic input space must all be one dimensional"
            combined_critic_input_size += input_space_shape[0]
        
        assert combined_actor_input_size > 0 and combined_critic_input_size > 0 , "The input space cannot be 0 in size"
        
        self.combined_actor_input_size = combined_actor_input_size
        self.combined_critic_input_size = combined_critic_input_size
                
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
                layers_actor.append(nn.Linear(actor_hidden_layer_sizes[i-1], size))
   
            layers_actor.append(self.activation_fn())
            
            # The last layer must take action size into account
            layers_actor.append(nn.Linear(actor_hidden_layer_sizes[-1], action_size))
        
        # Add a last tanh function to map to output space between -1 and 1    
        layers_actor.append(nn.Tanh())
        
        # the asterix is unpacking all layers_actor items and passing them into the nn.Sequential
        self.actor = nn.Sequential(*layers_actor).to(self.device)
        
        # create the critic network
        layers_critic = []
        if len(critic_hidden_layer_sizes) == 0:
            # Simple One Layer Network without hidden layers -> maps to one value of the state
            layers_critic.append(nn.Linear(combined_critic_input_size, 1))
        else:
            layers_critic.append(nn.Linear(combined_critic_input_size, critic_hidden_layer_sizes[0]))
            
            for i, size in enumerate(critic_hidden_layer_sizes):
                
                if i == 0:
                    continue
                
                # add the activation function to the layer
                layers_critic.append(self.activation_fn())
                
                # add the linear layer
                layers_critic.append(nn.Linear(critic_hidden_layer_sizes[i-1], size))
   
            layers_critic.append(self.activation_fn())
             
            layers_critic.append(nn.Linear(critic_hidden_layer_sizes[-1], 1))
            
        # the critic does not need a last layer
        
        # the asterix is unpacking all layers_critic items and passing them into the nn.Sequential
        self.critic = nn.Sequential(*layers_critic).to(self.device)
        
        self._build_optim()
        
    def predict_action(self, actor_obs: Dict[str, torch.Tensor]):
        """Create an action using the actor network and a gaussian distribution
        -> the tensors will become detached after execution (only use for actiung eith the env)
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
        
        
        
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        
        self.dist = Normal(action_mean, action_std)
        
        
        action = self.dist.sample()
         
        
        log_prob = self.dist.log_prob(action)
        
        
        return action.detach(), log_prob.detach()
        
    def evaluate(self, critic_obs: Dict[str, torch.Tensor]):
        """Evaluate a state observation via the critic model

        Args:
            critic_obs (Dict[str, torch.Tensor]): [description]
        """
        critic_concat_obs = None
        
        for key in critic_obs:
            
            if critic_concat_obs:
                torch.cat((critic_obs, critic_obs[key]), dim= 1)
            else:
                critic_concat_obs = critic_obs[key]
        
        return self.critic(critic_concat_obs)

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
        
        
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        
        self.dist = Normal(action_mean, action_std)
        
        actions = self.dist.sample()
        
        log_prob = self.dist.log_prob(actions)
                        
                        
        return actions, values, log_prob
        
    
    def evaluate_actions(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """
        Evaluate actions according to the current policy

        Args:
            actor_obs (Dict[str, torch.Tensor]): [description]
            critic_obs (Dict[str, torch.Tensor]): [description]
            actions (torch.Tensor): [description]
            
        Returns:
            estimated state value, log_prob of taking that action, entropy of the action distribution
        """
        
        # concat the extracted obs  to usable tensors
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
                
        action_mean = self.actor(actor_concat_obs)
        values = self.critic(critic_concat_obs)
        
        
        
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        
        self.dist = Normal(action_mean, action_std)
         
        
        log_prob = self.dist.log_prob(actions)
        dist_entropy = self.dist.entropy()
        
        
        return values, log_prob, dist_entropy
        
    def save(self, path: str) -> None:
        """Save the policy to the given path

        Args:
            path (str): [description]
        """
        
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        
    
    def load(self, path: str) -> None:
        """Load the policy from the given path

        Args:
            path (str): [description]
 
        """
        
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        