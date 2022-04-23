
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

from tonian.common.spaces import MultiSpace

class MultispaceNetElement(nn.Module):
    
    def __init__(self, name: str, inputs_names: List[str], net: nn.Sequential) -> None:
        """Element Network for the multispace network
        The multispace network is made up of multiple of MultispaceNetElements
        Examples would be a cnn, mlp 
        Args:
            name (str): name of the multispace net element, cannot collide with any obersvation name the multispace net is going to use
            inputs_names (List[str]): names of inputs for the network, that will be used and concativated together 
            net (nn.Sequential): Network, that takes the cumulative size of all inputs names as input
        """
        super().__init__()
        self.name = name
        self.input_names = inputs_names
        self.net = net
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultispaceNet(nn.Module):
    
    
    def __init__(self, network_layers: List[List[MultispaceNetElement]]) -> None:
        """Takes an multispace input dict as input for the foward function 

        Args:
            network_layers (List[List[MultispaceNetElement]]
                 consecutive layers the network is made of.
                    layer 0 only takes observations
                    layer 1 takes outputs from layer 0 and obsersvations
                    layer 2 takes outputs from layer 1 and layer 0 and observations
                    .... and so on...
        """
        super().__init__()
        self.network_layers = network_layers
        
        
    def forward(self, x: Dict[str, torch.Tensor]):
        
        
        for i_layer in range(len(self.network_layers)):
            
            for element in self.network_layers[i_layer]:
                # concat inputs
                input = None
                for input_name in element.input_names:
                    if input is None:
                        input = x[input_name]
                    else: 
                        input = torch.cat((input, x[input_name]), dim=1)
        
                if i_layer != len(self.network_layers) - 1:
                    x[element.name] = element(input)
                else: 
                    return element(input)
        
    def to(self, device):
        for network_layer in self.network_layers:
            for network in network_layer:
                network.to(device) 
        return self
        
