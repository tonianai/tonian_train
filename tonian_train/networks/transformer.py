

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
    
    def __init__(self, config: Dict) -> None:
        super().__init__()
        
        
class OutputEmbedding(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        
        

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerNetLogStd(nn.Module):
    
    def __init__(self, action_space: gym.spaces.Space, 
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
        
    