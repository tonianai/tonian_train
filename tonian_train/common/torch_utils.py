 # This file contains not jit version of jit helper torch tensor manipulations 
import torch  
from typing import Dict
    

 

 
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)
  
def batch_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the dot product for every tensor in the batch

    Args:
        a (torch.Tensor): Shape(batch_size, x)
        b (torch.Tensor): Shape(batch_size, x)

    Returns:
        torch.Tensor: dot_product
    """
    concat = torch.cat((torch.unsqueeze(a, dim=-1), torch.unsqueeze(b, dim= -1)), dim= -1)

    return torch.sum( torch.prod(concat, dim= -1), dim= -1)



def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)



def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)



@torch.jit.script
def shift_tensor(tensor: torch.Tensor, shift_amount: int ):
    if shift_amount == 0:
        return tensor
    
    orig_shape = list(tensor.shape)
    orig_shape[1] = shift_amount
    
    zero_padding = torch.zeros(orig_shape, device=tensor.device, dtype=tensor.dtype)
    tensor= tensor[:, shift_amount:]
    
    return torch.cat(tensors=(tensor, zero_padding), dim = 1)
    
    

 
@torch.jit.script
def repeated_indexed_tensor_shift(tensor: torch.Tensor, roll_tensor: torch.Tensor):
    """Shift a tensor by the roll amount to the left in the first dimension repeatedly
    
    tensor:             roll_tensor:            result:
    [[a,b,c,d],         [1, 2]                  [[b,c,d,0],
    [e,f,g,h],                          -->      [g,h,0,0]
    [i,j,k,l],                                   [j,k,l,0]
    [m,n,o,p]]                                   [o,p,0,0]]               
    
    
    Example:
    

    Args:
        tensor (torch.Tensor): _description_
        roll_tensor (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    
    
    assert tensor.shape[0] % roll_tensor.shape[0] == 0, "The roll tensor length should be a whole number divisor of tensor.shape[0] "
    
    tile_amount = tensor.shape[0] / roll_tensor.shape[0] # how ofthen the pattern needs to be applied 
    
    range_tensor = torch.arange(tile_amount, device=tensor.device, dtype=torch.int64)
    
    for i in range(roll_tensor.shape[0]):
        
        row_indices = i + (range_tensor * roll_tensor.shape[0])

        tensor[row_indices] = shift_tensor(tensor=tensor[row_indices], shift_amount=roll_tensor[i])
        
    return tensor
    

 

@torch.jit.script
def indexed_tensor_roll(tensor: torch.Tensor, roll_tensor: torch.Tensor, dim: int = 0 ):
    """Roll a tensor of shape (a,b,..) with roll tensor shape(a or b)  

    ATTENTION: Very SLOW!!!
    Args:
        tensor (torch.Tensor): multidomensopnal tensor
        roll_tensor (torch.Tensor): shift tensor dtype = int
        dim (_type_): int, dimension to shift

    Returns:
        _type_: _description_
    """
    assert roll_tensor.shape[0] == tensor.shape[dim] 
    
    for row in range(tensor.shape[dim]):
        shift_amount: int = int(roll_tensor[row].item())
        tensor[row] = torch.roll(tensor[row], shifts=[shift_amount], dims= [0])
        
    return tensor
  
  

def tensor_dict_clone(t_dict: Dict[str, torch.Tensor]):
    res_dict = {}
    for key in t_dict.keys():
        res_dict[key] = t_dict[key].clone().detach()
        
    return res_dict