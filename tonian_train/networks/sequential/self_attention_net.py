
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

import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    # implementation motivated by pytorch.or tutorials transformer_tutorial.html
    
    def __init__(self, d_model: int, dropout: float = 0.03, max_len: int = 5000, learnable=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1) * 0.1, requires_grad=learnable)  # Scale parameter, learnable if needed

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len , batch_size, embedding_dim]``
        """
        x = x +  self.scale * self.pe[:x.size(0)]
        #x = self.dropout(x) 
        return x
    
    


class SelfAttentionNet(SequentialNet):
    
    
    def __init__(self, 
                 obs_space: MultiSpace, 
                 action_space: gym.Space,
                 embedding: ObsEncoder, 
                 d_model: int,
                 num_heads: int,
                 num_encoder_layers=3,
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False,
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(),
                 value_size: int = 1
                 ) -> None:
        super().__init__()
        
        self.embedding = embedding
        self.pos_encoder = PositionalEncoding(d_model)
         
        # Define heads for action mean, action standard deviation, value, and optional next state prediction
        self.action_mu_head = nn.Linear(d_model, action_space.shape[0])
        self.value_head = nn.Linear(d_model, value_size)
        self.num_actions = action_space.shape[0]
        
        self.value_activation = value_activation
        self.std_activation = std_activation 
        self.action_activation = action_activation
         
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_encoder_layers
        )
        
        self.has_residual_dynamics = False
        
        self.is_std_fixed = is_std_fixed
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(
                self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(
                d_model, self.num_actions)
         
        
        
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                tgt_action_mu: torch.Tensor,  
                tgt_action_std: torch.Tensor,  
                tgt_value: torch.Tensor, 
                tgt_mask=None,
                src_pad_mask=None,
                tgt_pad_mask=None):
        
        src = self.embedding(src_obs, True)
        src = src.permute(1, 0, 2)
        # src shape: (sequence_length, batch_size, d_model)
        
        src = self.pos_encoder(src)
        
        latent = self.encoder(src, src_key_padding_mask=src_pad_mask)
        latent = src
        
        # Selecting only the last element of the sequence
        latent = latent[-1, :, :]  # New shape: (batch_size, d_model)
 
         
        mu = self.action_mu_head(latent)
    
        if self.is_std_fixed:
            std = self.std_activation(self.action_std)
        else:
            std = self.std_activation(self.action_std(latent)) 
            
        value = self.value_activation(self.value_head(latent))
        
        return mu, mu*0 + std, value, None
    
    def has_encoder(self) -> bool:
        return True
    
       
    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'network.pth'))


    
def build_self_attention_net(config: Dict, 
                             obs_space: MultiSpace, 
                             action_space: gym.Space) -> SelfAttentionNet:
    """_summary_

    Args:
        config (Dict): _description_
        obs_space (MultiSpace): _description_
        action_space (gym.Space): _description_

    Returns:
        SelfAttentionNet: _description_
    """
    network_config = config['network']
    
    sequence_length = config['sequence_length']
    has_embedding = 'encoder' in network_config

    obs_embedding = None
    if has_embedding:
        obs_embedding = ObsEncoder(network_config['encoder'], obs_space, sequence_length=sequence_length)
        if 'model_path' in network_config['encoder']:
            obs_embedding.load_pretrained_weights(network_config['encoder']['model_path'])
            
            if network_config.get('freeze_encoder', False):        
                obs_embedding.freeze_parameters()
        obs_space = obs_embedding.get_output_space()
    
    d_model = network_config['d_model']
    num_heads = network_config['num_heads']
    num_encoder_layers = network_config['num_encoder_layers']
    
    action_activation = ActivationConfiguration(
        network_config.get('action_activation', 'None')).build()
    std_activation = ActivationConfiguration(
        network_config.get('std_activation', 'None')).build()

    value_activation = ActivationConfiguration(
        network_config.get('value_activation', 'None')).build()
    
    value_size = network_config.get('value_size', 1)
    
    return SelfAttentionNet( obs_space = obs_space, 
                 action_space = action_space,
                 embedding = obs_embedding, 
                 d_model = d_model,
                 num_heads = num_heads,
                 num_encoder_layers= num_encoder_layers,
                 action_activation = action_activation,
                 is_std_fixed = True ,
                 std_activation = nn.Identity(),
                 value_activation = value_activation,
                 value_size = value_size
                 )
        