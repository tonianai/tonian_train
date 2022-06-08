from collections import defaultdict
from typing import Dict, Optional, Tuple , Union
from torch.utils.tensorboard import SummaryWriter
import os, torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
import json
 

# This logger is similar to the logger used in stable-baselines3


class BaseLogger(ABC):
    """
    The logger class
    Loggers can take in value to store or display (e.g Reward, avg_std, etc)

    
    """
    
    def __init__(self, folder: str, identifier: Union[int, str], ) -> None:
        # determines whether the run has been set to create a new run or to continue a run
        self.identifier = identifier
        self.run_assigned : bool = False 
        self.folder = folder
        self.save_folder = os.path.join(folder, 'saves')
        
        pass
     

    @abstractmethod
    def log(self, key: str, value: Union[int, float], step: int):
        pass
    
    
    @abstractmethod
    def log_config(self, config: Dict):
        pass
    

    def _pretty_config_dict(config: Dict):
        json_hp = json.dumps(config, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    
        
        
    

    
class DummyLogger(BaseLogger):
    """A logger, that essentially does nothing. Used when a logger is required, but not needed e.g for tests

    Args:
        BaseLogger (_type_): 
    """
    def __init__(self) -> None:
        super().__init__('', 0)
    
    def log(self, key: str, value: Union[int, float], step: int):
        pass
        
    def log_config(self, tag:str, config: Dict):
        pass
     
class TensorboardLogger(BaseLogger):
        
    def __init__(self, folder: str, identifier: Union[int, str], print_to_console: bool = True) -> None:
        super().__init__(folder, identifier)
        self.folder = folder
        self.writer = SummaryWriter(log_dir=os.path.join(folder, "logs"))
        self.print_to_console = print_to_console
        

    def log(self, key: str, value: Union[int, float], step: int, verbose: bool = False):
        self.writer.add_scalar(key, value, step)
        
        if self.print_to_console and verbose:
            print(f"{step}  -  {key}: {value} ")
            
    def add_graph(self, model: nn.Module, observation: torch.Tensor = None):
        self.writer.add_graph(model, observation)
        
    def log_image(self, tag: str, image: np.ndarray):
        self.writer.add_image(tag, image)
        
    def log_config(self, tag: str, config: Dict):
        """Log the cofnig of the run to the tesnorboard via text

        Args:
            config (Dict): COnfiguration Dict
        """
        
        self.writer.add_text(tag,BaseLogger._pretty_config_dict(config))
        
    def log_config_items(self, tag: str, config: Dict):
        """Logg all direct children of a dict as a config themselves

        Args:
            tag (str): parent tag
            config (Dict): config with children
        """
        
        for key, value in config.items():
            if isinstance(value, Dict):
                self.log_config(tag + '/' + key, value)
        