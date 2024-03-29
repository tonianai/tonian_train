

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import os, yaml

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
    

             
            
def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        os.makedirs(base_folder_name)
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base
        
     
def parseActvationFunction(string: str):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    return activations[string]
