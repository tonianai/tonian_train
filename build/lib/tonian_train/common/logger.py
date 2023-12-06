from collections import defaultdict
from typing import Dict, Optional, Tuple , Union, List
from torch.utils.tensorboard import SummaryWriter
import os, torch, csv
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
import json
import wandb
wandb.login()
 

# This logger is similar to the logger used in stable-baselines3


class BaseLogger(ABC):
    """
    The logger class
    Loggers can take in value to store or display (e.g Reward, avg_std, etc)

    
    """
    

    def __init__(self) -> None:
        super().__init__()
     

    @abstractmethod
    def log(self, key: str, value: Union[int, float], step: int):
        pass
    
    
    @abstractmethod
    def log_config(self, config: Dict):
        pass
    

    def _pretty_config_dict(config: Dict):
        json_hp = json.dumps(config, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    
    @abstractmethod 
    def update_saved(self):
        pass
    
    
class LoggerList(BaseLogger):
    """Contains multiple loggers and passes trough function calls

    Args:
        BaseLogger (_type_): _description_
    """
    
    def __init__(self, loggers: List[BaseLogger], identifier: int, folder: str) -> None:
        super().__init__()
        
        self.loggers = loggers
        self.identifier = identifier
        self.folder = folder
        
    
        
    def log(self, key: str, value: Union[int, float], step: int):
        [logger.log(key, value, step) for logger in self.loggers ]
        
        
    def log_config(self, config):
        [logger.log_config(config) for logger in self.loggers ]
        
    def update_saved(self):
        [logger.update_saved() for logger in self.loggers ]
    
    
        
        
    

    
class DummyLogger(BaseLogger):
    """A logger, that essentially does nothing. Used when a logger is required, but not needed e.g for tests

    Args:
        BaseLogger (_type_): 
    """
    def __init__(self) -> None:
        super().__init__()
    
    def log(self, key: str, value: Union[int, float], step: int):
        pass
        
    def log_config(self, tag:str, config: Dict):
        pass
    
    
    def log_image(self, tag: str, image: np.ndarray):
        pass
    
    def update_saved(self):
        pass
     
class TensorboardLogger(BaseLogger):
        
    def __init__(self, folder: str, identifier: Union[int, str], print_to_console: bool = True) -> None:
        super().__init__()
        self.identifier = identifier
        self.run_assigned : bool = False 
        self.save_folder = os.path.join(folder, 'saves')
        
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
                
    
    def update_saved(self):
        pass
    
                
class CsvFileLogger(BaseLogger):
    
    def __init__(self, folder: str, identifier: Union[int, str]) -> None:
        super().__init__()
        
        self.identifier = identifier
        
        self.file_path = os.path.join(folder, 'last_logs.csv')
        self.params = {}
        
    def log(self, key: str, value: Union[int, float], step: int):
        
        self.params[key] = value
        self.params['step'] = step
        
    def log_config(self, config: Dict):
        return super().log_config(config)
        
    def update_saved(self):
    
        with open(self.file_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(self.params.keys())
             
            writer.writerow(self.params.values())
            
            

                
class CsvMaxFileLogger(BaseLogger):
    
    def __init__(self, folder: str, identifier: Union[int, str]) -> None:
        super().__init__()
        
        self.identifier = identifier
        
        self.file_path = os.path.join(folder, 'max_logs.csv')
        self.params = {}
        
    def log(self, key: str, value: Union[int, float], step: int):
        
        self.params['step'] = step
        if key not in self.params or self.params[key] < value :
            self.params[key] = value
        
    def log_config(self, config: Dict):
        return super().log_config(config)
        
    def update_saved(self):
    
        with open(self.file_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(self.params.keys())
             
            writer.writerow(self.params.values())
            
            
class WandbLogger(BaseLogger):
    
    def __init__(self, identifier: Union[int, str],
                 project_name: str = "tonian") -> None:
        super().__init__()
        self.identifier = identifier
        self.project_name = project_name
        print("Wandb Logger initialized")
    
    def log(self, key: str, value: Union[int, float], step: int):
        assert self.run is not None, "the log_config must be called before the log function on the WandbLogger"
        wandb.log({key: value}, step= step)
        
    def log_config(self, tag:str, config: Dict):
        self.run = wandb.init(
            # Set the project where this run will be logged
            project=self.project_name,
            # Track hyperparameters and run metadata
            config= config,
            id= self.identifier
            )
    
    def update_saved(self):
        pass
    
    
        
        
            
        