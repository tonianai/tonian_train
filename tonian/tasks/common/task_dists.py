
from typing import Dict, Union, Tuple, Type, Any, Callable
from abc import ABC, abstractmethod 
import random

import numpy as np
import torch



class TaskDistribution(ABC, Callable):
    
    def __init__(self, config: Dict) -> None:
        """A distribtuion used to sample parameters from for the task environment 

        Args:
            config (Dict): Configuration Dict: Should contain at least {dist_type: str}
                            
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def sample(self)-> Union[str, float, int]:
        raise NotImplementedError()
    
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sample(*args, **kwds)

class TaskFixedDistribution(TaskDistribution):
    
    def __init__(self, value: Union[str, int, float]) -> None:
        """The fixed distribution does not sample a value from a dist, it just returns the value given in the 
        The TaskFixedDistribution is a sort of dummy task distribution
        Args:
            value (Union[str, int, float]): _description_
        """
        config = {'dist_type':'fixed', 'value': value}
        super().__init__(config)
        self.value = value
        
    def sample(self) -> Union[str, float]:
        return self.value
        
    
        
class TaskGaussianDistribution(TaskDistribution):
    
    def __init__(self, config: Dict) -> None:
        """Gaussian sample a value given a mean and an std
        
        Args:
            config (Dict): Example: {dist_type:'gaussian', mean: 0.0,  std: 1.0}
        """
        super().__init__(config)
        
        assert 'mean' in config, "Mean needs to be set in an gaussian distribution"
        assert 'std' in config, "The standard deviation (std) must be set for the task_gaussian distribution"
        
        self.mean = float(config['mean'])
        self.std = float(config['std']) 
        self.randomize = bool(config.get('randomize', True))
        
    def sample(self) -> float:
        if self.randomize:
            return np.random.normal(self.mean, self.std, (1,))[0]
        else:
            return self.mean

class TaskSelectionDistribution(TaskDistribution):
    
    def __init__(self, config: Dict) -> None:
        """Sample a values from a list of possible values

        Args:
            config (Dict): Example: {dist_type: 'selection', selections : ['selection1', 'selection2', 'selection3']}
        """
        super().__init__(config)
        
        assert 'selections' in config, "The keyword selections must be set for a TaskSelectionDistribution"
        
        self.selections = config['selections']
        
    def sample(self) -> Union[str, float, int]:
        random.sample(self.selections, 1)[0]        
        
# maps the string config name of a task_dist to the class type
dist_types: Dict[str, Type[TaskDistribution]] = {
    "gaussian": TaskGaussianDistribution,
    "selection": TaskSelectionDistribution
}


def task_dist_from_config(config_or_value: Union[Dict, Any]) -> TaskDistribution:
    """Create the task distribution from the 

    Args:
        config_or_value (Dict, Any): if the config is a dict, the correct task_dist will be created, if it is not a dict, a TaskFixedDistribution with the given value will be created 

    Returns:
        TaskDistribution: resulting distribution
    """
    
    if isinstance(config_or_value, Dict):
        assert 'dist_type' in config_or_value, "The config for a task distribution should contain the key dist type to set the type of the distribution"
        return dist_types[config_or_value['dist_type']](config_or_value)
    else:
        return TaskFixedDistribution(config_or_value)