
from typing import Dict,  Optional
from abc import ABC

import torch, gym, os, math
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.torch_utils import tensor_mul_along_dim

from tonian_train.networks.elements.network_elements import *
from tonian_train.networks.elements.simple_networks import A2CSimpleNet

from tonian_train.networks.sequential.base_seq_nn import SequentialNet, InputEmbedding, OutputEmbedding 

 
    
    
class SimpleSequentialNet(SequentialNet):
    
    def __init__(self, 
                 simple_net: A2CSimpleNet,
                 obs_embedding = None) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                                                          
                                                          
                              
                               |                                       |   
                               |                                       |                           
                              \|/                                     \|/      
             
                                    |------------------------------|
                                    |     Simpe forward Multispace |
                                    |------------------------------|

    
                                                 |
                                    _____________|__________  
                                    |                        |
                                    |                        |
                                    |                        |
                                    \|/                      \|/
                            
                            |---------------|         |---------------|        
                            |   Actor Head  |         |  Critic Head  |
                            |---------------|         |---------------|
                            
                                    |                        |
                                    |                        |
                                   \|/                      \|/
                                    
                                Actions (mu & std)        States
                                    

        Args:
            simple_net: A2CSimpleNet: This network is the main network that will be used to process the observations and produce the actions and values. It is not inherently sequential, but it will be used in a sequential manner
            
        """
        super().__init__()
    
        self.simple_net = simple_net
    
     
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor, 
                      tgt_mask=None,
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src_obs (Dict[str, torch.Tensor]): tensor shape (batch_size, src_seq_length, ) + obs_shape
            tgt_action_mu (torch.Tensor): shape (batch_size, tgt_seq_length, action_length)
            tgt_action_std (torch.Tensor):  (batch_size, tgt_seq_length, action_length)
            tgt_value (torch.Tensor): (batch_size, sr)
            tgt_mask (_type_, optional): _description_. Defaults to None.
            src_pad_mask (_type_, optional): _description_. Defaults to None.
            tgt_pad_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
         
         
        multiplicative_src_mask = -1* (src_pad_mask.to(torch.uint8) -1) 
        
        # mask the src for observations that do not belong to this sequence
        for key in src_obs.keys():
            src_obs[key] = tensor_mul_along_dim(src_obs[key], multiplicative_src_mask)
 
        # flatten the src
        for key in src_obs.keys():
            src_obs[key] = torch.flatten(src_obs[key], start_dim=1)
 
        return self.simple_net(src_obs)
    
         
    






