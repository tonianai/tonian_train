import imp
from typing import Union, Dict, Type
import torch
import torch.nn as nn


# Observations can eigther be a torch.Tensor or a dict of torch tensors
Observation = Union[torch.Tensor, Dict[Union[str, int], torch.Tensor]]



# There is ńo parent class for all activationclasses 
# this is a substitizte
ActivationFnClass = Type[nn.Module]