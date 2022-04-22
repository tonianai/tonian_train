
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

from tonian.common.spaces import MultiSpace



class MultispaceNet(nn.Module):
    """Takes an multispace iput dict as input 
    """
    
    
    def __init__(self, network_layers: List[Dict[str, nn.Sequential]]) -> None:
        super().__init__()
        
