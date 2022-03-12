import gym
import torch
import torch.nn as nn
import random
import numpy as np
import torch as th


from typing import Dict, Iterable, Optional, Tuple, Union, List


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device

def dict_to(torch_dict: Dict[str, torch.Tensor], device: str):
        for i in torch_dict:
            torch_dict[i].to(device)
        return torch_dict
    
def dict_to_cpu(torch_dict: Dict[str, torch.Tensor]):
    for i in torch_dict:
        torch_dict[i] = torch_dict[i].cpu()
    return torch_dict

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def join_configs(base_config: Dict, config: Dict) -> Dict:
    """Joins two configuration files into one

    Args:
        base_config (Dict): The base config
        config (Dict): This config can override values of the base config

    Returns:
        Dict: [description]
    """
    # idea go through all the values and join or override if the value is not a dict
    # if the value is a dict recursevely call this function
    
    final_dict = base_config.copy()
    
    for key, value in config.items():
        
        if isinstance(value, Dict):
            
            # check if it is in the base
            if key in final_dict.keys():
                # join the dicts using a recursive call
                final_dict[key] = join_configs(final_dict[key], value)
            else:
                final_dict[key] = value
            
        else:
            final_dict[key] = value
        
    
    return final_dict


                    
                     
            
            
        
        
        
        
        
        
            

    
    