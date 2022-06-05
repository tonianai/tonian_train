 
import torch 
from isaacgym.torch_utils import *


 


@torch.jit.script
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


@torch.jit.script
def get_batch_tensor_2_norm(a: torch.Tensor) -> torch.Tensor:
    """Calculate and return the two norm of a given tensor

    Args:
        a (torch.Tensor): input tensor shape(batch_size, x)

    Returns:
        torch.Tensor: two norm shape(batch_size)
    """
    return a.pow(2).sum(dim=1).sqrt()

@torch.jit.script
def batch_normalize_vector(a: torch.Tensor) -> torch.Tensor:
    """Normalize a vector as a batch 

    Args:
        a (torch.Tensor): shape(batch_size, x)

    Returns:
        torch.Tensor: the same tensor normalizes (batch_size, x)
    """
    return torch.einsum("ia, i -> ia", a, 1 / a.pow(2).sum(dim=1).sqrt())
 
 
@torch.jit.script
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

@torch.jit.script
def normalized_batch_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the normalized dot product for every tensor in the batch

    Args:
        a (torch.Tensor): Shape(batch_size, x)
        b (torch.Tensor): Shape(batch_size, x)

    Returns:
        torch.Tensor: dot_product
    """
     
    # Calculate the normalized vectors
    a = torch.einsum("ia, i -> ia", a, 1 / a.pow(2).sum(dim=1).sqrt())
    b = torch.einsum("ia, i -> ia", b, 1 / b.pow(2).sum(dim=1).sqrt())
    
    concat = torch.cat((torch.unsqueeze(a, dim=-1), torch.unsqueeze(b, dim= -1)), dim= -1)

    return torch.sum( torch.prod(concat, dim= -1), dim= -1)


@torch.jit.script
def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Calculate euler angles (Radians) from a quaternion tensor 
    
        
        0. roll is rotation around x in radians (counterclockwise)
        1. pitch is rotation around y in radians (counterclockwise)
        2. yaw is rotation around z in radians (counterclockwise)

    Args:
        quat (torch.Tensor): (x, y, z, w) shape (batch_size, 4)

    Returns:
        torch.Tensor: euler angles (x, y, z) : shape (batch_size, 3)
    """
    
    t0 = 2.0 * (quat[: ,3] * quat[: , 0] + quat[:, 1] * quat[: , 2])
    t1 = 1.0 - 2.0 * (quat[: ,0] * quat[: , 0] + quat[:, 1] * quat[: , 1])
    roll_x = torch.unsqueeze(torch.atan(t0, t1), 0)
    
    t0 = 2.0 * (quat[: ,3] * quat[: , 1] - quat[:, 2] * quat[: , 0])
    t0 = torch.clamp(t0, -1, 1)
    pitch_y = torch.unsqueeze(torch.asin(t0),0)
    
    t0 = 2.0 * (quat[: ,3] * quat[: , 2] + quat[:, 0] * quat[: , 1])
    t1 = 1.0 - 2.0 * (quat[: ,1] * quat[: , 1] + quat[:, 3] * quat[: , 3])
    yaw_z = torch.unsqueeze(torch.atan2(t0, t1), 0)
    
    return torch.stack((roll_x, pitch_y, yaw_z), 0)

    
    
    
    
    
    
    

