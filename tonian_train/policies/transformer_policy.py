

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
 
 

class SequenceBuffer():
    
    def __init__(self,
                 sequence_length: int, 
                 obs_space: MultiSpace,
                 action_space: gym.spaces.Space,
                 store_device: Union[str, torch.device] = "cuda:0",
                 out_device: Union[str, torch.device] = "cuda:0",
                 n_envs: int = 1, 
                 n_values: int = 1) -> None:
        """Buffer, that stores the observations and the outputs for the transformer sequence length
        
        Args:
            sequence_length (int): _description_
            actor_obs_space (MultiSpace): _description_
            action_space (gym.spaces.Space): _description_
            store_device (_type_, optional): _description_. Defaults to "cuda:0".
            out_device (_type_, optional): _description_. Defaults to "cuda:0".
            n_envs (int, optional): _description_. Defaults to 1. 
            n_values (int, optional): _description_. Defaults to 1.
        """
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        assert self.action_size, "Action size must not be zero"
        

        self.n_values = n_values
        self.sequence_length = sequence_length 
        self.obs_space = obs_space
        
        self.store_device = store_device
        self.out_device = out_device
        self.n_envs = n_envs
        
        pass
    
    def reset(self) -> None:
        """
        Create the buffers and set all initial values to zero
 
        """
        self.dones = torch.zeros(( self.n_envs, self.sequence_length,), dtype=torch.int8, device=self.store_device)
        self.values = torch.zeros((self.n_envs, self.sequence_length, self.n_values), dtype=torch.float32, device=self.store_device)
         
        # the mean of the action distributions
        self.action_mu = torch.zeros((self.n_envs, self.sequence_length, self.action_size), dtype=torch.float32, device=self.store_device)
        # the std(sigma) of the action distributions   
        self.action_std = torch.zeros((self.n_envs, self.sequence_length, self.action_size), dtype= torch.float32, device=self.store_device)
         
        
        self.obs = {}
        for key, obs_shape in self.obs_space.dict_shape.items():
            self.obs[key] = torch.zeros((self.n_envs, self.sequence_length) + obs_shape, dtype=torch.float32, device= self.store_device)
        
        self.src_key_padding_mask = torch.ones((self.n_envs, self.sequence_length), device= self.store_device)
        self.tgt_key_padding_mask = torch.ones((self.n_envs, self.sequence_length), device= self.store_device)
        
        
    def add(
        self, 
        obs: Dict[str, torch.Tensor],
        action_mu: torch.Tensor,
        action_std: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor):
        
        
        # roll the last to the first position
        
        # The tensord fill upd from 
        for key in self.obs:   
            self.obs[key] =  torch.roll(self.obs[key], shifts=(-1), dims=(1)) 
        
        self.action_mu = torch.roll(self.action_mu, shifts=(-1), dims=(1))
        self.action_std = torch.roll(self.action_std, shifts=(-1), dims=(1))
        self.values = torch.roll(self.values, shifts=(-1), dims=(1))
        self.dones = torch.roll(self.dones, shifts=(-1), dims=(1))
        
        for key in self.obs:   
            self.obs[key][:, 0] = obs[key].detach().to(self.store_device)
            
        self.action_mu[:, 0] = action_mu.detach().to(self.store_device)
        self.action_std[:, 0] = action_std.detach().to(self.store_device)
        self.values[:, 0] = values.detach().to(self.store_device)
        self.dones[:, 0] = dones.detach().to(self.store_device)
        
        # for every true dones at index 1 -> erase all old states to the left
        
        last_dones = self.dones[:, 1]
        
        self.action_mu[1::]
        
    def get(self):
        
        return self.
        
        
 
class TransformerPolicy(A2CBasePolicy):
    
    def __init__(self, transformer_net: TransformerNetLogStd, 
                       sequence_length: int,
                       obs_normalizer: Optional[RunningMeanStdObs] = None) -> None:
        super().__init__(transformer_net)
        self.transformer_net = transformer_net
        self.obs_normalizer = obs_normalizer
        self.sequence_length = sequence_length
         
        
    
    def forward(self, is_train: bool,   
                      actor_obs: Dict[str, torch.Tensor],  
                      prev_actions: Optional[torch.Tensor]) -> Dict:
        """

        Args:
            is_train: Determines whether the output of the policy will be used for training
            actor_obs (Dict[str, torch.Tensor]): actor_observations 
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
            
