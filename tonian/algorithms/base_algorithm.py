


from typing import Dict, Type

import torch 
import torch.nn as nn 
from tonian.tasks.base.base_env import BaseEnv
from tonian.common.logger import BaseLogger, TensorboardLogger
from tonian.common.utils.utils import join_configs
from typing import Union

from abc import abstractmethod, ABC

import os 

import yaml

class BaseAlgorithm(ABC):
    
    
    def __init__(self, env: BaseEnv,  
                 config: Dict, 
                 device: Union[str, torch.device], 
                 logger: BaseLogger) -> None:
        super().__init__()
        
        
        # merge the config with the standard condfig
        base_config = self._get_standard_config()
        self.config = join_configs(base_config=base_config, config=config)
        
        self.logger = logger
        self.env = env 
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
    
    
    @abstractmethod
    def _get_standard_config(self) -> Dict:  
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        raise NotImplementedError()
    