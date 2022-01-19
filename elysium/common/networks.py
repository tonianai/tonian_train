
from abc import abstractmethod, abstractproperty
from turtle import forward
import torch
import torch.nn as nn


from typing import Dict, Optional, Union, Tuple, Type

from elysium.common.spaces import MultiSpace
from elysium.common.type_aliases import Observation, ActivationFnClass

from abc import ABC, abstractmethod


class BaseNet(nn.Module, ABC):
    """Base network used for the policies  
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    
    @abstractmethod
    def forward(self, obs: Observation) -> torch.Tensor:
        pass
    
    @abstractproperty
    def latent_dim(self) -> int:
        """
            returns the dimensionality of the result of the forward pass. type int
        """
        pass
    
    
class SimpleDynamicForwardNet(BaseNet):
    
    def __init__(self,
                 obs_shapes: Union[Dict[ Union[str, int] ,Tuple[int, ...]], int, Tuple[int]],
                 hidden_layer_sizes : Tuple[int],
                 activation_fn_type: ActivationFnClass,
                 device: Union[str, torch.device]) -> None:
        """The SimpleDynamicForwardNet used for a flat input linearly to a output
        Args:
            obs_shapes (Tuple[Tuple[int, ...]]): Observation shapes of the inputs 
                if the obs consists only of a single tensor, the obs_shapes consits of a Tuple in multidim cases or an int in single dim cases
                if the obs consists of a Dict of tensors, the obs_shapes consits of a dict with the shape of the tensor as value and the same keys as the obs dict
            hidden_layer_sizes (Tuple[int]): [description]
        """
        
        super().__init__()
        self.obs_shapes = obs_shapes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        
        # set the combined obs size
        self.combined_obs_size = 0
        if type(obs_shapes) is Tuple:
            # check if the tuple is multidimensional 
            assert len(obs_shapes) == 1, "The obs shapes on the SimpleDynamicForwardNet must be one dimenstional"
            self.combined_obs_size = obs_shapes[0]

        elif type(obs_shapes) is int:
            self.combined_obs_size = obs_shapes
            
        else:
            # The type of te obs_shapes is a Dict -> add all linear layers together
            for key in self.obs_shapes:
                single_input_shape = self.obs_shapes[key]
                assert len(single_input_shape) == 1, "All obs shapes on the SimpleDynamicForwardNet must be one dimensional"
                self.combined_obs_size += single_input_shape[0]
                
        
        # the type of the activation
        self.activation_fn_type = activation_fn_type        
        
        assert self.combined_obs_size == 0, "The input space cannot be 0 in size on the SimpleDynamicForwardNet"
        
        
        # create the layers of the network
        layers = []
        if len(hidden_layer_sizes) != 0:
            
            # add the first linear linear layer starting with the combined obs size
            layers.append(nn.Linear(self.combined_obs_size, hidden_layer_sizes[0]))
            
            for i , size in enumerate(layers):
                
                if i == 0:
                    continue
                
                # add the activation function to the layer
                layers.append(self.activation_fn_type())

                # add the linear layer
                layers.append(nn.Linear(hidden_layer_sizes[i-1],size))
            
            layers.append(self.activation_fn_type())
            
            # NOTE: the last layer is a literal and not included in this network
           
        # the asterix is unpacking all layers_actor items and passing them into the nn.Sequential
        self.network = nn.Sequential(*layers).to(self.device)
         
        
        
    def forward(self, obs: Observation) -> torch.Tensor:
        """Takes in an observation (either, dict or tensor)
        and returns literal with the last hidden layers size
        Args:
            obs (Observation): [description]

        Returns:
            torch.Tensor: [description]
        """
        
        concat_obs = None
        
        if type(obs) is dict:
            # the dict has to be concatinated before it can be used as an 
            for key in obs:
                if concat_obs:
                    torch.cat((concat_obs, obs[key]), dim=1)
                else:
                    concat_obs = obs[key]
        else:
            concat_obs = obs
        
        return self.network(concat_obs)
    
    
    @property
    def latent_dim(self) -> int:
        """
            returns the dimensionality of the result of the forward pass. type int
        """
        return self.hidden_layer_sizes(-1)