
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional

from tonian.training2.networks import A2CBaseNet, A2CSequentialNetLogStd
from tonian.training2.common.aliases import ActivationFn, InitializerFn


import torch, gym
import torch.nn as nn
import numpy as np
 
class A2CBasePolicy(nn.Module, ABC):

    
    def __init__(self, a2c_net: A2CBaseNet) -> None:
        super().__init__()
        self.a2c_net = a2c_net
        
    @abstractmethod
    def is_rnn(self):
        raise NotImplementedError()
    
    def forward(self, is_train: bool,   actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor], prev_actions: Optional[torch.Tensor]) -> Dict:
        """

        Args:
            is_train: Determines whether the output of the policy will be used for training
            actor_obs (Dict[str, torch.Tensor]): actor_observations
            critic_obs (Dict[str, torch.Tensor]): critic observations, that are additional to the actor observations
            prev_actions: The previous actions taken, only relevant if is_train = True
        Returns:
            Dict: result_dict following keys: 
                if is_train: 'prev_neglogprob', 'values', 'entropy', 'mus', 'sigmas'
                else: 'neglogpacs', 'values', 'actions', 'mus', 'sigmas'
                    
        """
          
        
        raise NotImplementedError()
    

class A2CBasePolicy(A2CBasePolicy):
    
    def __init__(self, a2c_net: A2CBaseNet) -> None:
        super().__init__()
        self.a2c_net = a2c_net
        
    def is_rnn(self):
        return self.a2c_net
    
    
    def forward(self, is_train: bool,   actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor], prev_actions: Optional[torch.Tensor] = None) -> Dict:
        """

        Args:
            is_train: Determines whether the output of the policy will be used for training
            actor_obs (Dict[str, torch.Tensor]): actor_observations
            critic_obs (Dict[str, torch.Tensor]): critic observations, that are additional to the actor observations
            prev_actions: The previous actions taken, only relevant if is_train = True
        Returns:
            Dict: result_dict following keys: 
        """
        mu, logstd, value = self.a2c_net(actor_obs, critic_obs)
        
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogprob = self.neglogprob(prev_actions, mu, sigma, logstd)
            
            result = {
                    'prev_neglogprob' : torch.squeeze(prev_neglogprob),
                    'values' : value,
                    'entropy' : entropy, 
                    'mus' : mu,
                    'sigmas' : sigma
                }                
            return result
        else:
            selected_action = distr.sample()
            neglogprob = self.neglogprob(selected_action, mu, sigma, logstd)
            
            result = {
                    'neglogpacs' : torch.squeeze(neglogprob),
                    'values' : value,
                    'actions' : selected_action,
                    'mus' : mu,
                    'sigmas' : sigma
                }
            return result
                
    def neglogprob(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

        
        
        