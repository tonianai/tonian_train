
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional

from tonian.training2.common.networks import MultispaceNet
from tonian.training2.common.aliases import ActivationFn, InitializerFn


import torch, gym
import torch.nn as nn
 


 
class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def has_critic_obs(self):
        return False

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None
    
class A2CSequentialPolicyLogStd(BasePolicy):
    
    def __init__(self, shared_actor_net: MultispaceNet,
                       action_space: gym.spaces.Space,
                       action_activation: ActivationFn = nn.Identity(),
                       is_std_fixed: bool = False, 
                       std_activation: ActivationFn = nn.Identity(),
                       value_activation: ActivationFn = nn.Identity(),
                       value_size: int = 1,
                       critic_net: Optional[MultispaceNet] = None,
                       ) -> None:
        """ A2CPolicy with seperated actor_critic approach

            Note: in this approach the critic also has all the actor observations
                     
                     
                       |-----------------|         |--------------------|
        actor obs ->   | shared_actor_net| -->     | residual_actor_net | -> action_dist -> action
                       |-----------------| \       |--------------------|
                                            \
                       |------------|        \     |--------------------|
        critic_obs ->  | critic_net |   ->    ---> | residual_critic_net| -> value
                       |------------|              |--------------------|
        
        

        Args:
            shared_actor_net (MultispaceNet): Network, that takes in the actor_obs and is a predceding net for both actor and critic
            action_space (gym.spaces.Space): The space the output of the policy should conform to
            action_activation (ActivationFn): The activation function of the actions 
            is_std_fixed (bool): Determines whether the action stanadard deviation (aslo called sigma) is dependend on the output of the aciton
                - it is a parameter of the network eitherways
            critic_net (MultispaceNet): Network of the critic
            residual_actor_net (MultispaceNet): residual network of the actor
            residual_critic_net (MultispaceNet): residual netowrk of the crititic, that takes outputs from shared_actor_net and from critic_net
        """
        super().__init__()
        self.shared_actor_net = shared_actor_net
        self.critic_net = critic_net
        assert len(action_space.shape) == 1, 'Multidim actions are not yet supported'
        self.num_actions = action_space.shape[0]
        
        self._has_critic_obs = self.critic_net is not None
        self.residual_actor_net = torch.nn.Linear(shared_actor_net.out_size, self.num_actions)
        if self._has_critic_obs:
            self.residual_critic_net = torch.nn.Linear(shared_actor_net.out_size + shared_actor_net.out_size, value_size)
        self.action_space = action_space
        
        self.action_mu_activation = action_activation
        self.value_activation = value_activation
        
        self.is_continuous = isinstance(self.action_space, gym.spaces.Box)
        
        self.is_std_fixed = is_std_fixed
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(shared_actor_net.out_size, self.num_actions)
        
        assert self.is_continuous, "Non continuous action spaces are not yet supported in A2CSequentialPolicyLogStd"
        
        self.std_activation = std_activation
        
    def is_rnn(self):
        return False
    
    def has_critic_obs(self):
        return self._has_critic_obs
        
        
    def forward(self, actor_obs: Dict[torch.Tensor], critic_obs: Optional[Dict[torch.Tensor]] = None ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on a A2C sequential logstd policy

        Args:
            actor_obs (Dict[torch.Tensor]): Multispace Observation of the actor
            critic_obs (Optional[Dict[torch.Tensor]], optional): Multispace Observation of the critic, that are additional to the actor_ibs.
                Defaults to None.
        """
        
        a_out = self.shared_actor_net(actor_obs)
        
        
        if self._has_critic_obs:
            c_out = self.critic_net(critic_obs)
            
            critic_in = torch.cat((c_out, a_out))
            value = self.value_activation(self.critic_net(critic_in))
            
            
        else:
            a_out = a_out.flatten(1) # the batch must be preserved
            value = self.value_activation(self.critic_net(a_out))
            
        if self.is_continuous:
            mu = self.action_mu_activation(self.residual_actor_net(a_out))
            
            if self.is_std_fixed:
                std = self.std_activation(self.action_std)
            else:
                std = self.std_activation(self.action_std(a_out))
                
            # mu*0 + std uses the approptiate shape 
            return mu, mu*0 + std, value
                
            
            
            
            
        
        pass
    
        
        
        
        