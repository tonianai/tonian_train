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
    
    