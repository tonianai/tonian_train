

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional, List

from tonian_train.common.spaces import MultiSpace
from tonian_train.networks import A2CBaseNet, A2CSimpleNet, build_simple_a2c_from_config
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.running_mean_std import RunningMeanStdObs
from tonian_train.policies.base_policy import A2CBasePolicy


import torch, gym, os
import torch.nn as nn
import numpy as np
 

class A2CSimplePolicy(A2CBasePolicy):
    
    def __init__(self, a2c_net: A2CSimpleNet, 
                 obs_normalizer: Optional[RunningMeanStdObs] = None) -> None:
        super().__init__(a2c_net)
        self.a2c_net = a2c_net
        self.obs_normalizer = obs_normalizer
        

    def save(self, path):
        
        torch.save(self.a2c_net.state_dict() ,os.path.join(path, 'network.pth'))   
        
        if self.obs_normalizer:
            torch.save(self.obs_normalizer.state_dict(), os.path.join(path, 'obs_norm.pth'))
            
   
    def load(self, path):
        
        self.a2c_net.load_state_dict(torch.load(os.path.join(path, 'network.pth' )), strict=False)
        
        if self.obs_normalizer:
            self.obs_normalizer.load_state_dict(torch.load(os.path.join(path, 'obs_norm.pth')))
             
    
    def is_rnn(self):
        return self.a2c_net
    
    def _normalize_obs(self, 
                       obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Normalize The observations

        Args:
            obs (Dict[str, torch.Tensor]): Multispace observations
        """
        if self.obs_normalizer:
            obs = self.obs_normalizer(obs)
             
             
        return obs
        
    def forward(self, is_train: bool, obs: Dict[str, torch.Tensor], prev_actions: Optional[torch.Tensor] = None) -> Dict:
        """

        Args:
            is_train: Determines whether the output of the policy will be used for training
            obs (Dict[str, torch.Tensor]): observations
            prev_actions: The previous actions taken, only relevant if is_train = True
        Returns:
            Dict: result_dict following keys: 
        """
        with torch.no_grad():
            obs = self._normalize_obs(obs)
        
        mu, logstd, value = self.a2c_net(obs)
 
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
                
                
def build_a2c_simple_policy(config: Dict,
                                  obs_space: MultiSpace, 
                                  action_space: gym.spaces.Space):
    
    network = build_simple_a2c_from_config(config, obs_space, action_space)
    
    normalize_inputs = config.get('normalize_input', True)
    
    obs_normalizer = None
    
    if normalize_inputs:
        obs_normalizer = RunningMeanStdObs(obs_space.dict_shape)
        
            
    return A2CSimplePolicy(network, obs_normalizer)
        
        
        


        

        