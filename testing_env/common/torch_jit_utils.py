 
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

    
def convert_num_to_one_hot_tensor(x: torch.Tensor, num_states: int, device) -> torch.Tensor:
    """Convert a tensor with a batch of integer values to a tensor with one hot encoding

    Args:
        x (torch.Tensor): _description_
        num_states (int): _description_
    """
    conversion_map = torch.eye(num_states, device=device, dtype=torch.int16)
    return conversion_map[x.to(dtype=torch.int64)]   

@torch.jit.script
def save_angle_between_vectors(a: torch.Tensor, b: torch.Tensor):
    """calculate the angle between two vectors
     when the normalized value is -1 or 1, the 
    Args:
        a (torch.Tensor): Shape(batch_size, x (2 or 3))
        b (torch.Tensor): Shape(batch_size, x (2 or 3))

    Returns:
        torch.Tensor: angle in radients (batch_size, )
    """ 
     
    normalized_batch_product = normalized_batch_dot_product(a, b)
    normalized_batch_product = torch.clamp(normalized_batch_product, min=-0.999, max=0.999) 
    return torch.acos(normalized_batch_product)


@torch.jit.script
def get_euler_xyz(q):
    qw, qx, qy, qz = 3, 0,1,2
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)