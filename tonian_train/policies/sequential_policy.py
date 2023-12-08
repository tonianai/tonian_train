

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional, List

from tonian_train.common.spaces import MultiSpace
from tonian_train.networks import build_simple_a2c_from_config, SimpleSequentialNet
from tonian_train.common.aliases import ActivationFn, InitializerFn
from tonian_train.common.running_mean_std import RunningMeanStdObs
from tonian_train.policies.base_policy import A2CBasePolicy 
from tonian_train.networks.sequential.non_seq_wrapper import SequentialNetWrapper


import torch, gym, os
import torch.nn as nn
import numpy as np
  
class SequentialPolicy(A2CBasePolicy):
    
    def __init__(self, sequential_net: SimpleSequentialNet, 
                       sequence_length: int,
                       obs_normalizer: Optional[RunningMeanStdObs] = None) -> None:
        super().__init__(sequential_net)
        self.sequential_net = sequential_net
        self.obs_normalizer = obs_normalizer
        self.sequence_length = sequence_length
         
        
    
    def forward(self, is_train: bool,   
                      src_obs: Dict[str, torch.Tensor],  
                      tgt_action_mu: torch.Tensor,
                      tgt_action_std: torch.Tensor,
                      tgt_value: torch.Tensor,
                      src_padding_mask: torch.Tensor,
                      tgt_padding_mask: torch.Tensor,
                      prev_actions: Optional[torch.Tensor] = None,
                      tgt_mask: Optional[torch.Tensor] = None,) -> Dict:
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
        
            
        mu, logstd, value, next_state_pred = self.sequential_net.forward(src_obs, 
                                                         tgt_action_mu,
                                                         tgt_action_std, 
                                                         tgt_value,
                                                         tgt_mask=tgt_mask,
                                                         src_pad_mask=src_padding_mask,
                                                         tgt_pad_mask=tgt_padding_mask) # TODO: This must add the proper masks
 
        sigma = torch.exp(logstd)
   
        distr = torch.distributions.Normal(mu, sigma)
        
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogprob = self.neglogprob(prev_actions[:, -1], mu, sigma, logstd)
            
            result = {
                    'prev_neglogprob' : torch.squeeze(prev_neglogprob),
                    'values' : value,
                    'entropy' : entropy, 
                    'mus' : mu,
                    'sigmas' : sigma,
                    'next_state_pred' : next_state_pred
                }                
            return result
        else:
            selected_action = distr.sample()
            neglogprob = self.neglogprob(selected_action, mu, sigma, logstd)
            
            result = {
                    'neglogprobs' : torch.squeeze(neglogprob),
                    'values' : value,
                    'actions' : selected_action,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'next_state_pred' : next_state_pred
                }
            return result
                
    def neglogprob(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
            
    def get_tgt_mask(self, size) -> torch.tensor:
        return self.sequential_net.get_tgt_mask(size=size)

    def normalize_obs(self, 
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
            


def build_a2c_sequential_policy(config: Dict, obs_space: MultiSpace, action_space:gym.spaces.Space):
    
    
    
    sequence_length = config['sequence_length']
    network_type = config['network']['network_type']
    
    
    if network_type == 'simple_sequential':
        
        simple_a2c = build_simple_a2c_from_config(config['network'],
                                                  obs_space=obs_space,
                                                  action_space=action_space,
                                                  sequence_length= sequence_length)
        
        network = SimpleSequentialNet(simple_a2c)
        
        
    elif network_type == 'sequential_wrapper':
        
        simple_a2c = build_simple_a2c_from_config(config['network'],
                                                  obs_space=obs_space,
                                                  action_space=action_space)
        network = SequentialNetWrapper(simple_a2c)
    else:
        raise 'network_type not supported'


    normalize_inputs = config.get('normalize_inputs', True)
    
    if normalize_inputs:
        
        obs_normalizer = RunningMeanStdObs(obs_space.dict_shape, is_sequence=True)
        
    return SequentialPolicy( sequential_net= network,
                             sequence_length= sequence_length,
                             obs_normalizer= obs_normalizer
                             )