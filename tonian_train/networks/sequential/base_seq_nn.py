

from typing import Dict,  Optional
from abc import ABC

import torch, gym, os, math
import torch.nn as nn
import numpy as np
  
from tonian_train.networks.elements.network_elements import * 
 

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
            
            # TODO: THe error could be here
            assert obs_tensor.shape[1] == self.sequence_length , "The second dim of data sequence tensor must be equal to the sequence length +1"   
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
        print("pos encoder init")
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class SequentialNet(ABC, nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
