import torch
from torch import nn 
import torch.nn as nn
import numpy as np
from envs.base.vec_task import MultiSpace, VecTask, Env

from typing import Dict, Any, Tuple, Union

import yaml
import os

class WalkingTask(VecTask):
    
    def __init__(self, config_path: str, sim_device: str, graphics_device_id: int, headless: bool) -> None:
         
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"File {config_path} not found")
      
        #extract params from config 
        self.randomize = config["task"]["randomize"]
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
    
    def _get_input_spaces(self) -> MultiSpace:
        return super()._get_input_spaces()
    
    def _get_output_spaces(self) -> MultiSpace:
        return super()._get_output_spaces()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        return super().reset()
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return super().step(actions)
    
        
        
        