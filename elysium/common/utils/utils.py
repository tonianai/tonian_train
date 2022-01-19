import gym
import torch
import torch.nn as nn


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



                    
                     
            
            
        
        
        
        
        
        
            

    
    