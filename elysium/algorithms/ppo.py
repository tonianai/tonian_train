from typing import Dict, Union

import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal

from elysium.algorithms.base_algorithm import BaseAlgorithm
from elysium.tasks.base.vec_task import VecTask

from elysium.algorithms.policies import ActorCriticPolicy




class PPO(BaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, policy : ActorCriticPolicy) -> None:
        super().__init__(env, config)
        self._fetch_config_params(config)
        self.actor_obs_shapes = env.actor_observation_spaces.shape
        self.critic_obs_shapes = env.critic_observation_spaces.shape
        
    def _fetch_config_params(self, config):
        """Fetćh hyperparameters from the configuration dict and set them to member variables
        
        Args:
            config ([type]): [description]
        """
        
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.k_epochs = config['k_epochs']
        
    
        
    def train(timesteps: int):
        return super().train()
    
    
    def save(self, path: str):
        return super().save(path)


    def load(self, path: str):
        """Load from a given checkpoint
        Args:
            path (str): [description]
        """
        pass
    
    