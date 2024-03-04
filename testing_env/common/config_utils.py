

from typing import Optional

from tonian_train.common.logger import BaseLogger
from gym.spaces import space  
from tonian_train.common.spaces import MultiSpace
from testing_env.base_env.vec_task import  VecTask
from testing_env.running_test.mk1_running_task import  Mk1RunningTask
from testing_env.cartpole.cartpole_task import Cartpole

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import os, yaml

from typing import Optional
 

def task_from_config(config: Dict, headless: bool = False, seed: int = 0) -> VecTask:
    
    name_to_task_map=  { 
     "00_cartpole": Cartpole,
     "02_mk1_running": Mk1RunningTask
    }
    return name_to_task_map[config["name"]](config, sim_device="cuda", graphics_device_id=0, headless= headless)



