import torch
from torch import nn 
import torch.nn as nn
import numpy as np
from envs.base.vec_task import MultiSpace, VecTask, Env

from typing import Dict, Any, Tuple, Union


class WalkingTask(VecTask):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool) -> None:
        
        #extract params from config 
        self.randomize = self.config["task"]["randomize"]
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        
        
        
        