
import torch.nn as nn
import torch
from typing import Callable, Dict
from tonian.training2.common.types import ActivationFn, InitializerFn


def ActivationFunctionFactory( activation_fn_str: str, **kwargs) -> ActivationFn:
    """Creates a activation function given the activation function string and keyword arguments

    Args:
        activation_fn_str (str): String encoding the activation function

    Returns:
        ActivationFn: the instance of the acitvation function
    """
    activation_fn_class_map = {
        'relu' : nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid' : nn.Sigmoid,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'softplus': nn.Softplus,
        'None': nn.Identity
    }
    
    return activation_fn_class_map[activation_fn_str](**kwargs)


def InitializerFactory(initializer_fn_str: str, **kwargs) -> InitializerFn:
    """Creates an initializer function with the kwargs set

    Args:
        initializer_fn_str (str): String encoding of the initializer

    Returns:
        InitializerFn: Initializer function
    """
    def _create_initializer(func, **kwargs):
        return lambda v : func(v, **kwargs)   
    
    intializer_fn_class_map = {
        'const_initializer': _create_initializer(nn.init.constant_, **kwargs),
        'orthogonal_initializer': _create_initializer(nn.init.orthogonal_, **kwargs),
        'glorot_normal_initializer': _create_initializer(nn.init.xavier_normal_, **kwargs),
        'glorot_uniform_initializer': _create_initializer(nn.init.xavier_uniform_, **kwargs), 
        'random_uniform_initializer': _create_initializer(nn.init.uniform_, **kwargs),
        'kaiming_normal': _create_initializer(nn.init.kaiming_normal_, **kwargs),
        'orthogonal': _create_initializer(nn.init.orthogonal_, **kwargs),
        'default' : nn.Identity()
    }
    
    return intializer_fn_class_map[initializer_fn_str]
    
def PolicyFactory(config: Dict):
    """Creates a Policy given a config dict

    Args:
        config (Dict): 
    """
    
    

    
    