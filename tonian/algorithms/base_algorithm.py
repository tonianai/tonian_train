


from typing import Dict, Optional, Type
from tonian.tasks.base.vec_task import VecTask

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
    
    
    def __init__(self, env: BaseEnv, config: Dict = {}, device: Union[str, torch.device] = "cuda:0", logger: Optional[BaseLogger] = None) -> None:
        super().__init__()
        
        
        # merge the config with the standard condfig
        base_config = self._get_standard_config()
            
        self.env = env
        self.config = join_configs(base_config=base_config, config=config)
        self.device = device

        # the name where  the network and log files will be stored about this run
        run_base_folder= f'runs/{type(env).__name__}'
        
        self.run_index = BaseAlgorithm.get_run_index(run_base_folder)
        
        self.run_folder_name = run_base_folder + "/"+ str(self.run_index)
         
        # create the run folder
        os.makedirs(self.run_folder_name)
        # create the run saves folder
        os.makedirs(self.run_folder_name + "/saves")
        # create the run logs folder
        os.makedirs(self.run_folder_name + "/logs")
        # save the config in the run folder
        
        self.save_config()
        
        
        if logger is None:
            self.logger = TensorboardLogger(self.run_folder_name + "/logs")
        else:
            self.logger = logger
        
        
        
    def get_run_index(base_folder_name: str) -> int:
        """get the index of the run
        Args:
            base_folder_name (str): The base folder all the runs are stored in 
        """
        if not os.path.exists(base_folder_name):
            os.makedirs(base_folder_name)
            
        n_folders_in_base = len(os.listdir(base_folder_name))
        
        return n_folders_in_base
        
    def save_config(self):
        """
            Save the config of the env and the algorithm in the folder of the run
        """
        
        with  open(f"{self.run_folder_name}/config.yaml", "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=True)
        
        with  open(f"{self.run_folder_name}/env_config.yaml", "w") as outfile:
            yaml.dump(self.env.config, outfile, default_flow_style=True)
        
        
    

    
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
    