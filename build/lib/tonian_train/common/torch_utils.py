 # This file contains not jit version of jit helper torch tensor manipulations 
import torch  
    

 

 
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