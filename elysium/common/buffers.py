from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple


import numpy as np
import torch



class BaseBuffer(ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def sample(self, num_sample: int):
        pass
    
    @abstractmethod 
    def clear(self):
        pass
    
class FixSizeBuffer(BaseBuffer):
    
    def __init__(self, buffer_len: int) -> None:
        super().__init__()
        self.buffer_len = buffer_len
        
    
    