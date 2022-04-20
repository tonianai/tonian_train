from typing import Callable, Dict, Union, List, Any
import torch.nn as nn
from abc import ABC, abstractmethod

InitializerFn = Callable
ActivationFn = nn.Module


class DictConfigurationType(ABC):
    
    def __init__(self, config: Dict) -> None:
        """A Configuration Type is a Python object representation of values entered into the config dict
        
                
        The build function is capable of creating the object instance of the object articulated in the dict
        All Arguments in the config Dict are the arguments for the creation, that are set at configuration time and can be set in a config.yaml for examle
        All Arguments passed in the build function are the arguments that are only set at runtime
        
        Args:
            config (Union[List, Dict])
        """
        super().__init__()
        
    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

class ActivationConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.name = config['name']
        self.kwargs = config.get('args', {} )
        
    def build(self) -> ActivationFn: 
        activation_fn_class_map = {
        'relu' : nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid' : nn.Sigmoid,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'softplus': nn.Softplus,
        'None': nn.Identity
        }
    
        return activation_fn_class_map[self.name](**self.kwargs) 

class InitializerConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.name = config['name']
        
        self.kwargs = config.get('args', {})
    
    def build(self) -> InitializerFn:
        
        def _create_initializer(func, **kwargs):
            return lambda v : func(v, **kwargs)   
    
        intializer_fn_class_map = {
            'const_initializer': _create_initializer(nn.init.constant_, **self.kwargs),
            'orthogonal_initializer': _create_initializer(nn.init.orthogonal_, **self.kwargs),
            'glorot_normal_initializer': _create_initializer(nn.init.xavier_normal_, **self.kwargs),
            'glorot_uniform_initializer': _create_initializer(nn.init.xavier_uniform_, **self.kwargs), 
            'random_uniform_initializer': _create_initializer(nn.init.uniform_, **self.kwargs),
            'kaiming_normal': _create_initializer(nn.init.kaiming_normal_, **self.kwargs),
            'orthogonal': _create_initializer(nn.init.orthogonal_, **self.kwargs),
            'default' : nn.Identity()
        }
    
        return intializer_fn_class_map[self.name]

class CnnLayer(DictConfigurationType):
    
    
    def __init__(self, config: Dict) -> None:
        """Configuration for a single CNN Layer

        Args:
            config (Dict): Example:
                filters: 32
                kernel_size: 8
                strides: 4
                padding: 0

        """
        super().__init__()
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.strides = config['strides']
        self.padding = config['padding']
    
    def build(self) -> Any:
        # TODO: Implement
        return super().build()

class CnnConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        
        self.type = config['type']
        self.activation = ActivationConfiguration(config['activation'])
        self.initializer = InitializerConfiguration(config['initializer'])
        
        self.layers = [CnnLayer(layer_config) for layer_config in config['convs']]
        
    def build(self) -> nn.Sequential:
        # TODO: Implement 
        return super().build()
    
class MlpConfiguration(DictConfigurationType):
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        
        self.units : List[int] =  config['units']
        self.activation = ActivationConfiguration(config['activation'])
        self.initializer = InitializerConfiguration(config['intitializer'])
        
    def build(self, input_size: int) -> nn.Sequential:
        pass
        