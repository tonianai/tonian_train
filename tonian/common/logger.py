from collections import defaultdict
from typing import Dict, Optional, Tuple , Union
from torch.utils.tensorboard import SummaryWriter
import os

from abc import ABC, abstractmethod

from tonian.policies.policies import BasePolicy

# This logger is similar to the logger used in stable-baselines3


class BaseLogger(ABC):
    """
    The logger class
    Loggers can take in value to store or display (e.g Reward, avg_std, etc)

    
    """
    
    def __init__(self, folder: str) -> None:
        # determines whether the run has been set to create a new run or to continue a run
        self.run_assigned : bool = False 
        self.folder = folder
        self.save_folder = os.path.join(folder, 'saves')
        
        pass
     

    
    
    @abstractmethod
    def log(self, key: str, value: Union[int, float], step: int):
        pass
    

    
    

    
     
class TensorboardLogger(BaseLogger):
    
    def __init__(self, folder: str, print_to_console: bool = True) -> None:
        super().__init__(folder)
        self.folder = folder
        self.writer = SummaryWriter(log_dir=os.path.join(folder, "logs"))
        self.print_to_console = print_to_console
        
        
     

           

    def log(self, key: str, value: Union[int, float], step: int, verbose: bool = False):
        self.writer.add_scalar(key, value, step)
        
        if self.print_to_console and verbose:
            print(f"{step}  -  {key}: {value} ")
            
