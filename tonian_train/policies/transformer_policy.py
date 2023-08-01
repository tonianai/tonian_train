

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional, List

from tonian_train.common.spaces import MultiSpace
from tonian_train.networks import A2CBaseNet, A2CSimpleNet, build_simple_a2c_from_config
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.running_mean_std import RunningMeanStdObs
from tonian_train.policies.base_policy import A2CBasePolicy
from tonian_train.networks import TransformerNetLogStd


import torch, gym, os
import torch.nn as nn
import numpy as np
 
 
 
class TransformerPolicy(A2CBasePolicy):
    
    def __init__(self, transformer_net: TransformerNetLogStd, obs_normalizer: Optional[RunningMeanStdObs] = None) -> None:
        super().__init__(transformer_net)
        self.transformer_net = transformer_net
        self.obs_normalizer = obs_normalizer
        
    
    def forward(self, is_train: bool,   
                      actor_obs: Dict[str, torch.Tensor], 
                      critic_obs: Dict[str, torch.Tensor],
                      prev_actions: Optional[torch.Tensor]) -> Dict:
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
        
        # This function must provide the correct masks and it must tile the padding for episodes, that just began
        
        with torch.no_grad():
            obs = self._normalize_obs(obs)
            
        
        
          
        
        raise NotImplementedError()

    def _normalize_obs(self, 
                       obs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Normalize The observations

        Args:
            obs (Dict[str, torch.Tensor]): Multispace observations
        """
        if self.obs_normalizer:
            obs = self.obs_normalizer(obs)
             
             
        return obs
    def is_rnn(self):
        return False
    
    def load(self, path):
        self.a2c_net.load_state_dict(torch.load(os.path.join(path, 'network.pth' )), strict=False)
        
        if self.obs_normalizer:
            self.obs_normalizer.load_state_dict(torch.load(os.path.join(path, 'obs_norm.pth')))
             

    def save(self, path): 
        torch.save(self.a2c_net.state_dict() ,os.path.join(path, 'network.pth'))   
        
        if self.obs_normalizer:
            torch.save(self.obs_normalizer.state_dict(), os.path.join(path, 'obs_norm.pth'))
            
