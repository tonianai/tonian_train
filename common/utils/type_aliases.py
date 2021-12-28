from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

from abc import ABC, abstractmethod

import numpy as np
import torch 

    
    
# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
class Schedule:
    
    def __init__(self, start: float, end: float, update_by: float, update_every:int) -> None:
        
        self.start = start
        self.end = end
        self.update_by = update_by
        self.update_every = update_every
        
        self.current_val = start
        
    def update_maybe(self, i_step: int):
        
        if self.update_every % i_step == 0:
            self.update()
            
    def update(self):
        
        if self.end > self.start:
            # increase
            
            self.current_val += self.update_by
            
            if self.current_val > self.end:
                self.current_val = self.end
            
            
        elif self.end < self.start:
            
            #decay
            self.current_val -= self.update_by
            
            if self.current_val < self.end:
                self.current_val = self.end
            
    
class Space(ABC):
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def is_continuous(self):
        raise NotImplementedError
    
    
class InputSpace(Space):
    
    def __init__(self) -> None:
        super().__init__()        
        
        
        
class ContinuousInputSpace(InputSpace):
    
    def __init__(self, linear_obs_size: int, visual_obs_shape: Tuple[int, int, int] = None, command_size: int = 0) -> None:
        super().__init__()
        self.linear_obs_size = linear_obs_size
        self.visual_obs_shape = visual_obs_shape
        self.command_size = command_size
        pass
    
    def is_continuous(self):
        return True
    
    def has_visual(self):
        return self.visual_obs_shape != None
    
    def has_command(self):
        return self.command_size == 0
    
    def get_channel_size(self):
        if self.has_visual():
            return self.visual_obs_shape[0]
        return 0 
    
class OutputSpace(InputSpace):
    
    def __init__(self) -> None:
        super().__init__()
        
class ContinuousOutputSpace(OutputSpace):
    
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size
        
    def is_continuous(self):
        return True
    