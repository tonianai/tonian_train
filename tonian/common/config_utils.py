

from typing import Callable, Dict, Tuple

from tonian.common.logger import BaseLogger
from tonian.tasks.walking.walking_task import WalkingTask
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1.mk1_running.mk1_running_task import Mk1RunningTask 
from tonian.tasks.mk1.mk1_controlled.mk1_controlled import Mk1ControlledTask 
from tonian.tasks.mk1_multi_task.mk1_multi_task import Mk1Multitask
from tonian.tasks.mk1_multi_task_box.mk1_multi_task_box import Mk1MultitaskBox
from tonian.tasks.mk1.mk1_controlled_terrain.mk1_terrain import Mk1ControlledTerrainTask
from tonian.tasks.mk1.mk1_controlled_visual.mk1_visual import Mk1ControlledVisualTask

from gym.spaces import space 
from tonian.tasks.base.vec_task import  VecTask
from tonian.common.schedule import Schedule 
from tonian.common.spaces import MultiSpace

import torch
import torch.nn as nn
import os, yaml

from typing import Optional


     

def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        raise FileNotFoundError()
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base

                    
def parseActvationFunction(string: str):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    return activations[string]


def task_from_config(config: Dict, headless: bool = False) -> VecTask:
    
    name_to_task_map=  {
     "00_cartpole": Cartpole,
     "01_walking": WalkingTask,
     "02_mk1_running": Mk1RunningTask, 
     "03_mk1_controlled": Mk1ControlledTask,
     "04_mk1_multitask": Mk1Multitask,
     "05_mk1_multitask_box": Mk1MultitaskBox,
     "06_mk1_controlled_terrain": Mk1ControlledTerrainTask,
     "07_mk1_controlled_visual": Mk1ControlledVisualTask
    }
    return name_to_task_map[config["name"]](config, sim_device="cuda", graphics_device_id=0, headless= headless)




def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        os.makedirs(base_folder_name)
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base
        
    

def create_new_run_directory(config: Dict, batch_id: Optional[str] = None) -> Tuple[str, int]:
    """Create a new run directory and store the given config in the directory
    Args:
        config (Dict): The config file, that contains all the important info to recreate, or continue this run
         
 
    Returns:
        str: run_folder_name
    """
    
    task_name = config['task']['name']
    
    # the name where  the network and log files will be stored about this run
    run_base_folder= f'runs/{task_name}'
    
    if batch_id is not None:
        run_index = get_run_index(os.path.join(run_base_folder, batch_id))
        
        run_folder_name = os.path.join(run_base_folder, batch_id, str(run_index))
        
    else: 
        run_index = get_run_index(run_base_folder)
        run_folder_name = os.path.join(run_base_folder , str(run_index))
     
    # create the run folder
    os.makedirs(run_folder_name)
    # create the run saves folder
    os.makedirs(run_folder_name + "/saves")
    # create the run logs folder
    os.makedirs(run_folder_name + "/logs")
    # save the config in the run folder
    
    with  open(f"{run_folder_name}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=True)
        
    return (run_folder_name, run_index)
    

             
            