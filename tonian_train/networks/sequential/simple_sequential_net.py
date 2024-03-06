
from typing import Dict,  Optional
from abc import ABC

import torch, gym, os, math
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.torch_utils import tensor_mul_along_dim

from tonian_train.networks.elements.network_elements import *
from tonian_train.networks.elements.a2c_networks import A2CSimpleNet

from tonian_train.networks.sequential.base_seq_nn import SequentialNet, InputEmbedding, OutputEmbedding 
from tonian_train.networks.elements.encoder_networks import ObsEncoder
from tonian_train.networks.elements.a2c_networks import build_simple_a2c_from_config

 
    
    
class SimpleSequentialNet(SequentialNet):
    
    def __init__(self, 
                 simple_net: A2CSimpleNet,
                 obs_embedding = Optional[ObsEncoder]) -> None:
        """_summary_
                                                        
                                                        
                                                          
               Observations:                                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           
                                                       
                              
                               |                                   
                               |                                                            
                              \|/                                    
                              
                    |------------------|                  
                    | Input Embeddings |  (Optional)                              
                    |------------------|                                
                               |                                                         
                              \|/            
                              
                [..., org_latent(t-2)
                org_latent(t=1),
                org_latent(t=0)]         
             
                                                        |---------------------|
                                                    --> |   res_dynamics_net  | -> next_state (opt)
                                                    |   |---------------------|
                                                    |
                                                    | 
                                |------------|         |--------------------|
                 org_latent ->  |  a2csimple | -->  -- | residual_actor_net | -> action_dist -> action
                                |------------|         |--------------------|
                                                    |
                                                    |  |--------------------|
                                                    |->| residual_critic_net| -> value
                                                        |--------------------|
                                    

        Args:
            simple_net: A2CSimpleNet: This network is the main network that will be used to process the observations and produce the actions and values. It is not inherently sequential, but it will be used in a sequential manner
            
        """
        super().__init__()
    
        self.simple_net = simple_net
        self.obs_embedding = obs_embedding
    
     
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
        
        # optional embedding of the src_obs
        if self.obs_embedding:
            new_src_obs = {}
            new_src_obs['embedding'] = self.obs_embedding(src_obs, True) # src obs shape for each (batch_size, src_seq_length, ) + obs_shape
            src_obs = new_src_obs
            # src obs['embedding'] shape (batch_size, src_seq_length, d_model)
        multiplicative_src_mask = -1* (src_pad_mask.to(torch.uint8) -1) 
        
        # mask the src for observations that do not belong to this sequence
        for key in src_obs.keys():
            src_obs[key] = tensor_mul_along_dim(src_obs[key], multiplicative_src_mask)
 
        # flatten the src
        for key in src_obs.keys():
            src_obs[key] = torch.flatten(src_obs[key], start_dim=1)
 
        return self.simple_net(src_obs)
    
         
    



def build_simple_sequential_net(config: Dict[str, Any], obs_space: MultiSpace, action_space: gym.spaces.Space) -> SimpleSequentialNet:
    
    sequence_length = config['sequence_length']
    
    network_config = config['network']
    
    has_embedding = 'encoder' in network_config
    
    if has_embedding:
        obs_embedding = ObsEncoder(network_config['encoder'], obs_space, sequence_length=sequence_length)
        obs_embedding.load_pretrained_weights(network_config['encoder']['model_path'])
        obs_embedding.freeze_parameters()
        obs_space = obs_embedding.get_output_space()
        
    
    a2c_net = build_simple_a2c_from_config(network_config, obs_space, action_space, sequence_length)
    
    return SimpleSequentialNet(a2c_net, obs_embedding)
    

