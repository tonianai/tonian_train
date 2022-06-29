
from typing import Dict, Union, Tuple, Type, Any, Callable, Set
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

def get_kwarg_names_for_dist_type(dist_type: str) -> Set:
    """Retreives the keys for a distribution type

    Args:
        dist_type (str): string of the dist type (gaussian , selection ...)

    Returns:
        Set: All the kwargs needed for this distribution
    """
    
    dist_type_keys = {
        'gaussian': ['mean', 'std'],
        'selection': ['values']
    }
    
    return dist_type_keys[dist_type]
    


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
     
    
    
    
def sample_tensor_gaussian(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.normal(mean=mean, std=std)
    
def sample_tensor_selection(selections: torch.Tensor) -> torch.Tensor:
    """
    Args:
        selections (torch.Tensor): expected shape (num_selctions, possible_selections)
    
        # linear_drawn
        dist_type: selection
        values: [0,0,1,2,2, 2, 3,4]
        
        # The example would sample from a random point on the x axis from this plot and take the 
                        _
        4             _
        3      ______
        2    _  
        1____
            0 1 2 3 4 5 6 7   

    Returns:
        torch.Tensor: result shape(num_selections)
    """
    
    
    rand_index  = torch.round(torch.rand(selections.shape[0], device=selections.device) * (selections.shape[1] - 0.1 ) - 0.49).to(dtype=torch.int64)
    
    return selections[:, rand_index]
    print(rand_index)
    
    
     
    
def sample_tensor_uniform_dist(dist_config_or_value: Union[Dict, float, int], sample_shape: Tuple[int, ...], device: Union[str, torch.device] ):
    """Directly sample from a given distribution 
    The distribution only has single input values as a input
    following dist_config_or_value shapes are acceptable:
    
    value: int or float returns a torch tensor with the shape and on the device, that are given
    
    dist:
        # gaussian
            mean: float
            std: float or int
            dist_type: gaussian
            
            
            
        # linear_drawn
            dist_type: linear_drawn
            values: [0,0,1,2,2, 2, 3,4]
            
            # The example would sample from a random point on the x axis from this plot and take the 
            4              /
            3        ___ /
            2      /
            1____/
             0 1 2 3 4 5 6 7   
             
            
        # linear_drawn
            dist_type: selection
            values: [0,0,1,2,2, 2, 3,4]
            
            # The example would sample from a random point on the x axis from this plot and take the 
                            _
            4             _
            3      ______
            2    _  
            1____
             0 1 2 3 4 5 6 7   
             
             
    
            

    Args:
        dist_config_or_value (Dict): _description_
        sample_shape (Tuple[int, ...]): _description_
        device (Union[str, torch.device]): _description_
    """
    
    if not isinstance(dist_config_or_value, Dict):    
        return torch.ones(sample_shape, device= device) * dist_config_or_value
     
    dist_config = dist_config_or_value
    
    distribution_type = dist_config['dist_type']
    if distribution_type == 'gaussian':
        mean = torch.ones(sample_shape, device=device) * dist_config['mean']
        std = torch.ones(sample_shape, device=device) * dist_config['std']
        return torch.normal(mean = mean, std = std).unsqueeze(dim=1)
    
    if distribution_type == 'linear_drawn':
        values = dist_config['values']
        x_sample = (torch.rand(*sample_shape, device= device) * (len(values) - 1))
        value_torch = torch.tensor(values, device= device)
        lower = value_torch[x_sample.to(torch.int64)]
        upper = value_torch[torch.ceil(x_sample).to(torch.int64)]
        ratio = x_sample - torch.floor(x_sample)
         
        return (1 - ratio) * lower + ratio * upper
    
    if distribution_type == 'selection':
        values = dist_config['values']
        x_samples = torch.randint(low = 0 , high= (len(values) -1), size=sample_shape, device= device).to(torch.int64)
        value_torch = torch.tensor(values, device= device)
        return value_torch[x_samples]
        
        
        
        
        
        
    