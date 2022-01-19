import imp
from typing import Union, Dict, Union
import torch


Observation = Union[torch.Tensor, Dict[Union[str, int], torch.Tensor]]