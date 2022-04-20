
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any

import torch
import torch.nn as nn

 
class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def is_separate_critic(self):
        return False

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None
    
class A2CPolicyLogStd(BasePolicy):
    
    def __init__(self, network: ) -> None:
        super().__init__()
         