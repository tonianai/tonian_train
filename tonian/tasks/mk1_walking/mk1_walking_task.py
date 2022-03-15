from gym.spaces import space
import numpy as np
from tonian.tasks.base.command import Command
from tonian.tasks.base.vec_task import VecTask, BaseEnv, GenerationalVecTask


from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace

from gym import spaces
import gym

from typing import Dict, Any, Tuple, Union, Optional

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import yaml
import time
import os


import torch
from torch import nn 
import torch.nn as nn

class WalkingTask(GenerationalVecTask):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device) -> None:
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
            
    def _extract_params_from_config(self) -> None:
        return super()._extract_params_from_config()
    
    def _get_standard_config(self) -> Dict:
        return super()._get_standard_config()
    
    def _create_envs(self, num_envs: int, spacing: float, num_per_row: int) -> None:
        return super()._create_envs(num_envs, spacing, num_per_row)
    
    def pre_physics_step(self, actions: torch.Tensor):
        return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        return super().post_physics_step()
    
    def reset_envs(env_ids: torch.Tensor) -> None:
        return super().reset_envs()
    
    def _is_symmetric(self):
        return False
    