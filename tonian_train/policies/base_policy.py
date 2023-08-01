
from abc import ABC, abstractmethod
from typing import Dict, Optional

from tonian_train.networks import A2CBaseNet 


import torch, gym, os
import torch.nn as nn
import numpy as np
 
class A2CBasePolicy(nn.Module, ABC):

    
    def __init__(self, a2c_net: A2CBaseNet) -> None:
        super().__init__()
        self.a2c_net = a2c_net
        
    @abstractmethod
    def is_rnn(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load(self, path):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path):
        raise NotADirectoryError()
    
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
    