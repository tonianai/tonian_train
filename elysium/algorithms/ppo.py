from typing import Dict, Union

import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal

from elysium.algorithms.base_algorithm import BaseAlgorithm
from elysium.common.utils.utils import Schedule
from elysium.tasks.base.vec_task import VecTask

from elysium.algorithms.policies import ActorCriticPolicy




class PPO(BaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, policy : ActorCriticPolicy, device: str) -> None:
        super().__init__(env, config)
        self._fetch_config_params(config)
        self.actor_obs_shapes = env.actor_observation_spaces.shape
        self.critic_obs_shapes = env.critic_observation_spaces.shape
        
        
        self.buffer_size = self.env.num_envs * self.n_step
        
        assert self.batch_size > 1, "Batch size must be bigger than one"
        
        assert self.buffer_size > 1, "Buffer size must be bigger than one"
        
        assert self.buffer_size % self.batch_size == 0, "the buffer size must be a multiple of the batch size"
        
        
        
        
    def _fetch_config_params(self, config):
        """Fetćh hyperparameters from the configuration dict and set them to member variables
        
        Args:
            config ([type]): [description]
        """
        
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.n_epochs = config['k_epochs']
        self.batch_size = config['batch_size']
        self.n_step = config['n_step']
        self.target_kl = config['target_kl']
        
        self.action_std_schedule = Schedule(config['astion_std'])
        self.action_std = self.action_std_schedule(0)
    
        
    
    def train(self)-> None:
        """Update the policy using the memory buffer
        -> clear the buffer afterwards
        """
        
        
    
    def learn(self, 
              total_timesteps: int,
              log_inteval: int = 1, 
              ):
        
        
        return super().learn()
    
    
    def save(self, path: str):
        return super().save(path)


    def load(self, path: str):
        """Load from a given checkpoint
        Args:
            path (str): [description]
        """
        pass
    
    