

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
    
    def __init__(self, config: Dict, sequence_length: int ) -> None:
        """
        The goal of this network is to order the Observations in a ordered latent space
        
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
        assert config.has_key('encoder'), "Input Embeddings needs a encoder specified in the config"
        assert config['encoder'].has_key('network'), "The Input Embedding encoder needs a specified network. (List with mlp architecture)"
        
        self.network: MultispaceNet = MultiSpaceNetworkConfiguration(config['encoder']['network'])
        
        
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
            
        unstructured_result = self.network(unstructured_obs_dict)
        
        # restructuring, so that the output is (batch_size, sequence_length, ) + output_dim
        return unstructured_result.reshape((batch_size, self.sequence_length,)  +  unstructured_result.shape[1::])
    
    
    
        
        
        
class OutputEmbedding(nn.Module):
    
    def __init__(self, config: Dict, sequence_length: int) -> None:
        """
        The goal of this network is to order the actions (mu & std), values in a ordered latent space

        Args:
            config (Dict): _description_
        """
        super().__init__()
        
        
        
        

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length:int , dropout_p: float=  0.1 , max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, sequence_length, 2) * (-math.log(10000.0) / sequence_length))
        pos_encoding = torch.zeros(max_len, 1, sequence_length)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerNetLogStd(nn.Module):
    
    def __init__(self, 
                 action_space: gym.spaces.Space, 
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False, 
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(),
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
<Multipsace Net>--> | Input Embeddings |                      | Output Embedding |  <-- <Multispace Net>            
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
        
    