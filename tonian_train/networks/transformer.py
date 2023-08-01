

from typing import Callable, Dict, Union, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch, gym, os, math
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.spaces import MultiSpace

from tonian_train.networks.network_elements import *



class InputEmbedding(nn.Module):
    
    def __init__(self, config: Dict, 
                       sequence_length: int,
                       d_model: int, 
                       obs_space: MultiSpace) -> None:

        """
        The goal of this network is to order the Observations in a ordered latent space, ready for self attention 
        
        Example
        config: {
                encoder: 
                    network: [ <see conifg documentation of  MlpConfiguration> 
                        { 
                            name: linear_obs_net # this can be custom set
                            input: 
                                - obs: linear
                            mlp:
                                units: [256, 128]
                                activation: relu,
                                initializer: default
                        }
                    ]  
            >
            }

        Args:
            config (Dict): _description_
        """
        super().__init__()
        self.sequence_length = sequence_length
        assert 'encoder' in config, "Input Embeddings needs a encoder specified in the config"
        assert 'network' in config['encoder'], "The Input Embedding encoder needs a specified network. (List with mlp architecture)"
        
        self.network: MultispaceNet = MultiSpaceNetworkConfiguration(config['encoder']['network']).build(obs_space)
        
        self.d_model = d_model
        self.out_nn: nn.Linear = nn.Linear(self.network.out_size(), d_model )
        
    def forward(self, obs: Dict[str, torch.Tensor]):
        """_summary_

        Args:
            obs (Dict[str, torch.Tensor]): any tensor has the shape (batch_size, sequence_length, ) + obs.shape

        Returns:
            _type_: _description_
        """
        
        # TODO:  validate, that this is the right approach 
        # Note: Do we have a multiplication of gradients with this approach???
        # Please investigate @future schmijo
           
        unstructured_obs_dict = {} # the unstructuring has to happen, because the self.network only has one batch dimension, and here we essentially have two (batch, sequence_length) and would like to have one 
        for key, obs_tensor in obs.items():
                
            batch_size = obs_tensor.shape[0]
            assert obs_tensor.shape[1] == self.sequence_length, "The second dim of data sequence tensor must be equal to the sequence length"   
            unstructured_obs_dict[key] = obs_tensor.view((obs_tensor.shape[0] * obs_tensor.shape[1], ) + obs_tensor.shape[2::])
            
        unstructured_result = self.network(unstructured_obs_dict) # (batch_size * sequence_length, ) + output_dim
        
        # restructuring, so that the output is (batch_size, sequence_length, ) + output_dim
        return self.out_nn(unstructured_result).reshape((batch_size, self.sequence_length, self.d_model) )
    
     
class OutputEmbedding(nn.Module):
    
    def __init__(self, config: Dict, 
                       sequence_length: int, 
                       d_model: int, 
                       action_space: gym.spaces.Space,
                       value_size: int = 1) -> None:
        """
        The goal of this network is to order the actions (mu & std), values in a ordered latent space

        Args:
            config (Dict):
            
                encoder:
                    mlp: 
                        units: [256, 128]
                        activation: relu,
                        initializer: default
        """
        super().__init__()
        
        print(config)
        self.sequence_length = sequence_length
        assert 'encoder' in config, "Output Embeddings needs a encoder specified in the config"
        assert 'mlp' in config['encoder'], "The Output Embedding encoder needs a specified mlp architecture (key = mlp)."
        
        assert isinstance(action_space, gym.spaces.Box), "Only continuous action spaces are implemented at the moment"
        
        
        self.action_size = action_space.shape[0]
        self.value_size = value_size
        self.mlp_input_size = self.action_size *2 + self.value_size
        
        mlp_config = MlpConfiguration(config['encoder']['mlp'])
        self.network: nn.Sequential = mlp_config.build(self.mlp_input_size)
        
        
        self.d_model = d_model
        self.out_nn: nn.Linear = nn.Linear(mlp_config.get_out_size(), d_model )
        
        
            
    def forward(self, action_mu: torch.Tensor, action_std: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Embedding for outputs 
        Transforms outputs in a fitting latent space for self attention

        Args:
            action_mu (torch.Tensor): shape (batch_size, seq_len, action_size)
            action_std (torch.Tensor):  shape (batch_size, seq_len, action_size)
            value (torch.Tensor): shape (batch_size, seq_len, value_size)

        Returns:
            torch.Tensor: _description_
        """
        
        all_outputs = torch.cat((action_mu, action_std, value), 2)
        
        assert all_outputs.shape[2] == self.mlp_input_size, "The mlp input size does not fit the concatinated mlp input in the output embeddings"
        
        
        # TODO:  validate, that this is the right approach 
        # Note: Do we have a multiplication of gradients with this approach???
        # Please investigate @future schmijo
           # the unstructuring has to happen, because the self.network only has one batch dimension, and here we essentially have two (batch, sequence_length) and would like to have one
        unstructured_all_outputs = all_outputs.view((
            all_outputs.shape[0] *  all_outputs.shape[1], all_outputs.shape[2]
        ))
        
        result: torch.Tensor = self.network(unstructured_all_outputs)
    
        return self.out_nn(result).reshape(all_outputs.shape[0], all_outputs.shape[1], self.d_model)
       

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int , dropout_p: float=  0.1 , max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pos_encoding = torch.zeros(max_len, 1, d_model)
        self.pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', self.pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerNetLogStd(nn.Module):
    
    def __init__(self, 
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 n_heads: int,
                 input_embedding: InputEmbedding,
                 output_embedding: OutputEmbedding,
                 d_model: int,
                 sequence_length: int,
                 action_space: gym.spaces.Space, 
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False, 
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(),
                 pos_encoder_dropout_p: float = 0.1,
                 dropout_p: float = 0.1,
                 value_size: int = 1) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                              
                               |                                       |   
                               |                                       |                           
                              \|/                                     \|/      
                              
                    |------------------|                      |------------------|     
<Multipsace Net> -> | Input Embeddings |                      | Output Embedding |  <- <MLP>            
                    |------------------|                      |------------------|            
                               |                                       |                           
                              \|/           Organized Latent Space    \|/            
             
|------------|            |--------|                              |--------|          |------------| 
|Pos Encoding|    ->      |    +   |                              |    +   |     <-   |Pos Encoding| 
|------------|            |--------|                              |--------|          |------------|      

                               |                                       |                           
                              \|/                                     \|/            


                      |---------------|                      |---------------|
                      |               |                      |               |
                      | Encoder Block |  x N           |---->| Decoder Block |  x N
                      |               |                |     |               |
                      |---------------|                |     |---------------|
                              |                        |                            
                              |                        |             |
                              |________________________|             |
                                                                     |
                                                        _____________|__________  
                                                        |                        |
                                                        |                        |
                                                        |                        |
                                                       \|/                      \|/
                                                
                                               |---------------|         |---------------|        
                                               |   Actor Net   |         |   Critic Net   |
                                               |---------------|         |---------------|
                                               
                                                        |                        |
                                                        |                        |
                                                       \|/                      \|/
                                                       
                                                     Actions                  States
                                                       

        Args:
            action_space (gym.spaces.Space): _description_
            action_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            is_std_fixed (bool, optional): _description_. Defaults to False.
            std_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_size (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
        
        self.model_type = "Transformer"
        
        self.positional_encoder = PositionalEncoding(
            d_model=d_model , dropout_p=pos_encoder_dropout_p, max_len=5000
        )
        
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.d_model = d_model
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead= n_heads, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        
        
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor,  
                      tgt_mask=None, 
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src (torch.Tensor): (batch_size, src_seq_length, encdoding_length)
            tgt (torch.Tensor): (batch_size, src_seq_length, encoding_length)
        
        """
        
        
        src = self.input_embedding(src_obs) * math.sqrt(self.d_model)
        tgt = self.output_embedding(tgt_action_mu, tgt_action_std, tgt_value) * math.sqrt(self.d_model)
         
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
    
    
    
def build_transformer_a2c_from_config(config: Dict,
                                      seq_len: int, 
                                      value_size: int, 
                                      obs_space: MultiSpace,
                                      action_space: gym.spaces.Space) -> TransformerNetLogStd:
    """Build a transformer model for Advantage Actor Crititc Policy

    Args:
        config (Dict): config
        obs_space (MultiSpace): Observations
        action_space (gym.spaces.Space): Actions Spaces

    Returns:
        TransformerNetLogStd: _description_
    """
    
    d_model = config['d_model']

    
    input_embedding = InputEmbedding(config['input_embedding'], 
                                     sequence_length= seq_len,
                                     d_model= d_model,
                                     obs_space= obs_space)
    
    output_embedding = OutputEmbedding(config=config['output_embedding'],
                                       sequence_length= seq_len,
                                       d_model= d_model,
                                       action_space= action_space,
                                       value_size= value_size )
    
    
    
    
    
    
    