from collections import defaultdict
from typing import Dict, Optional, Tuple , Union
from torch.utils.tensorboard import SummaryWriter

from abc import ABC, abstractmethod

# This logger is similar to the logger used in stable-baselines3


class BaseLogger(ABC):
    """
    The logger class
    
    """
    
    def __init__(self) -> None:
         pass
     
    
    @abstractmethod
    def log(self, key: str, value: Union[int, float], step: int):
        pass
    
    
     
class TensorboardLogger(BaseLogger):
    
    def __init__(self, folder: str, print_to_console: bool = True) -> None:
        super().__init__()
        self.folder = folder
        self.writer = SummaryWriter(log_dir=folder)
        self.print_to_console = print_to_console
        

    def log(self, key: str, value: Union[int, float], step: int, verbose: bool = False):
        self.writer.add_scalar(key, value, step)
        
        if self.print_to_console and verbose:
            print(f"{step}  -  {key}: {value} ")
            
