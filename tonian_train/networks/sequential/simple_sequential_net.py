
from typing import Dict,  Optional
from abc import ABC

import torch, gym, os, math
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.torch_utils import tensor_mul_along_dim

from tonian_train.networks.network_elements import *
from tonian_train.networks.simple_networks import A2CSimpleNet

from tonian_train.networks.sequential.base_seq_nn import SequentialNet, InputEmbedding, OutputEmbedding 

 
    
    
class SimpleSequentialNet(SequentialNet):
    
    def __init__(self, 
                 main_body: nn.Sequential, 
                 target_seq_len: int,  
                 action_space: gym.spaces.Space, 
                 main_out_size: int, 
                 action_head: Optional[nn.Sequential] = None,
                 action_head_size: Optional[int] = None,
                 critic_head: Optional[nn.Sequential] = None,
                 critic_head_size: Optional[int] = None,
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False, 
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(), 
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
            action_space (gym.spaces.Space): _description_
            action_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            is_std_fixed (bool, optional): _description_. Defaults to False.
            std_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_activation (ActivationFn, optional): _description_. Defaults to nn.Identity().
            value_size (int, optional): _description_. Defaults to 1.
        """
        super().__init__()
    
     
        self.main_body = main_body 
        

        self.action_head = action_head
        self.critic_head = critic_head
        self.num_actions = action_space.shape[0]
         
        self.a_out = nn.Linear(action_head_size, self.num_actions)
      
            
        self.c_out = nn.Linear(critic_head_size, value_size)
        
        
        self.is_std_fixed = is_std_fixed
        
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(main_out_size, self.num_actions)
        
        
        self.action_activation = action_activation
        self.std_activation = std_activation
        self.value_activation = value_activation
    
     
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
        
        # mask the src and the tgt
        for key in src_obs.keys():
            src_obs[key] = torch.flatten(tensor_mul_along_dim(src_obs[key], multiplicative_src_mask), 1)
 

        main_out = self.main_body(src_obs) 
        
        
        value = self.value_activation(self.c_out(self.critic_head(main_out)))
        
        mu = self.action_activation(self.a_out(self.action_head(main_out)))
        
        if self.is_std_fixed:
            std = self.std_activation(self.action_std)
        else:
            std = self.std_activation(self.action_std(main_out))
        
        
        return mu, mu*0 + std, value      
    
         
    

def build_simple_sequential_nn_from_config(config: Dict,
                                      seq_len: int, 
                                      value_size: int, 
                                      obs_space: MultiSpace,
                                      action_space: gym.spaces.Space) -> SimpleSequentialNet:
    """Build a transformer model for Advantage Actor Crititc Policy

    Args:
        config (Dict): config
        obs_space (MultiSpace): Observations
        action_space (gym.spaces.Space): Actions Spaces

    Returns:
        EmbeddingSequentialNet: _description_
    """
     
     
    
    space_dict = {}
    for key, space in obs_space:
        shape = list(space.shape)
        shape[0] = shape[0] * (seq_len +1)
        shape = tuple(shape)  
        space_dict[key] = gym.spaces.Box(low= -1, high= 1, shape = shape)
        
    expanded_sequnced_obs_space = MultiSpace(space_dict)

    network: MultispaceNet = MultiSpaceNetworkConfiguration(config['main_body']).build(expanded_sequnced_obs_space)
      
    
    action_activation = ActivationConfiguration(config.get('action_activation', 'None')).build()
    std_activation = ActivationConfiguration(config.get('std_activation', 'None')).build()
    value_activation = ActivationConfiguration(config.get('value_activation', 'None')).build()
    
    action_head_factory = MlpConfiguration(config['action_head'])
    action_head = action_head_factory.build(network.out_size())
    
    critic_head_factory =  MlpConfiguration(config['critic_head'])
    critic_head = critic_head_factory.build(network.out_size())
     
    
    is_std_fixed = config['is_std_fixed']
    
    
    sequental_nn = SimpleSequentialNet( 
            main_body= network, 
            target_seq_len= seq_len, 
            action_space=action_space,
            action_head=action_head,
            action_head_size = action_head_factory.get_out_size(),
            critic_head=critic_head,
            critic_head_size = critic_head_factory.get_out_size(),
            main_out_size = network.out_size(),
            action_activation=action_activation,
            is_std_fixed= is_std_fixed,
            std_activation=std_activation,
            value_activation=value_activation,
            dropout_p= 0.1,
            value_size=value_size
    )
    
    return sequental_nn





