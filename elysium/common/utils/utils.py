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



class Schedule: 
    def __init__(self, schedule: Dict) -> None:
        """A schedule takes progess as an input (eg. steps, episodes, generations) and returns a value for a hyperparameter (e.g lr)
        Args:
            schedule (Dict): often defined in confing files:
            in yaml the dict should have the following structure:
            {
                schedule_type: 'steps', (or generations, reward etc)
                interpolation: 'linear', (or none -> results in step function)
                schedule: 
                    [0, 0.5],
                    [1e3, 0.4],
                    [2e3, 0.3],
                    [3e3, 0.2],
                    [1e4, 0.1]
                    
            }
        """
        self.schedule_type: str = schedule['schedule_type'] 
        self.schedule: Union[List[List[float]], List[Tuple[int]]] = schedule['schedule']
        
        #schedules with no values will be rejected
        assert len(self.schedule) != 0, "The schedule must have at least one value"
        
        
    def __call__(self, query: Union[float, int]):
        """Get a value for a query value -> This should not be executed too often, because it is not trivial to compute
        Args:
            query (Union[float, int]): [description]
        """
        
        # get the lowest query value 
        
      
        
        if query > self.schedule[-1][0]:
            # query value is below smalles schedule value -> no extrapolation
            return self.schedule[-1][1] 
        
        elif query < self.schedule[0][0]:
            
            # query value is above biggest schedule value -> no extrapolation
            return self.schedule[0][1]
        else:  
            
            lower = 0
            upper = 0
            
            for i in range(len(self.schedule) - 1):
                if self.schedule[i][0] <= query and self.schedule[i+1][0] >= query:
                    return self.schedule[i][1]
                    # Todo: add interpolation
                    
                     
            
            
        
        
        
        
        
        
            

    
    