
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional, List

from tonian.common.spaces import MultiSpace

from tonian.training.networks import A2CBaseNet, A2CSequentialNetLogStd, build_A2CSequientialNetLogStd
from tonian.training.common.aliases import ActivationFn, InitializerFn
from tonian.training.common.running_mean_std import RunningMeanStdObs


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
    

class A2CSequentialLogStdPolicy(A2CBasePolicy):
    
    def __init__(self, a2c_net: A2CSequentialNetLogStd, 
                 actor_obs_normalizer: Optional[RunningMeanStdObs] = None,
                 critic_obs_normalizer: Optional[RunningMeanStdObs] = None) -> None:
        super().__init__(a2c_net)
        self.a2c_net = a2c_net
        self.actor_obs_normalizer = actor_obs_normalizer
        self.critic_obs_normalizer = critic_obs_normalizer
        
        
    def is_rnn(self):
        return self.a2c_net
    
    def _normalize_obs(self, 
                       actor_obs: Dict[str, torch.Tensor], 
                       critic_obs: Optional[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Normalize The observations

        Args:
            actor_obs (Dict[str, torch.Tensor]): Multispace Actor observations
            critic_obs (Dict[str, torch.Tensor]): Multispace Critic observations
        """
        if self.actor_obs_normalizer:
            actor_obs = self.actor_obs_normalizer(actor_obs)
            
        if self.critic_obs_normalizer and critic_obs:
            critic_obs = self.critic_obs_normalizer(critic_obs)
            
            
        return actor_obs, critic_obs
        
        
    
    def forward(self, is_train: bool,   actor_obs: Dict[str, torch.Tensor], critic_obs: Optional[Dict[str, torch.Tensor]], prev_actions: Optional[torch.Tensor] = None) -> Dict:
        """

        Args:
            is_train: Determines whether the output of the policy will be used for training
            actor_obs (Dict[str, torch.Tensor]): actor_observations
            critic_obs (Dict[str, torch.Tensor]): critic observations, that are additional to the actor observations
            prev_actions: The previous actions taken, only relevant if is_train = True
        Returns:
            Dict: result_dict following keys: 
        """
        with torch.no_grad():
            actor_obs, critic_obs = self._normalize_obs(actor_obs=actor_obs, critic_obs=critic_obs)
        
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
                
                
def build_A2CSequentialLogStdPolicy(config: Dict,
                                  actor_obs_space: MultiSpace, 
                                  critic_obs_space: Optional[MultiSpace], 
                                  action_space: gym.spaces.Space):
    
    network = build_A2CSequientialNetLogStd(config, actor_obs_space, critic_obs_space, action_space)
    
    normalize_inputs = config.get('normalize_input', True)
    
    actor_obs_normalizer = None
    critic_obs_normalizer = None
    
    if normalize_inputs:
        
        actor_obs_normalizer = RunningMeanStdObs(actor_obs_space.dict_shape)
        
        if critic_obs_space:
            
            critic_obs_normalizer = RunningMeanStdObs(critic_obs_space.dict_shape)
            
    return A2CSequentialLogStdPolicy(network, actor_obs_normalizer, critic_obs_normalizer)
        
        
        


        

        