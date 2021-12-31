


from typing import Dict
from elysium.tasks.base.vec_task import VecTask

import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
from elysium.tasks.base.base_env import BaseEnv

from typing import Union

from abc import abstractmethod, ABC

class BaseAlgorithm(ABC):
    
    
    def __init__(self, env: BaseEnv, config: Dict, device: Union[str, torch.device]) -> None:
        super().__init__()
        self.env = env
        self.config = config
        self.device = device
    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()
    
    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError()
    
    