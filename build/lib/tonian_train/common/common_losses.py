from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional
import torch

def critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    if clip_value:
        value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.max(value_losses,
                                         value_losses_clipped)
    else:
        c_loss = (return_batch - values)**2
    return c_loss


def actor_loss(old_action_log_probs_batch, action_log_probs, advantage, is_ppo, curr_e_clip):
    if is_ppo:
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_log_probs * advantage)
    
    return a_loss



def calc_dynamics_loss(obs_prediction: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor]):
    """
    Calculate the loss between the predicted observation and the actual next observation for each component in the dictionaries.
    The function then returns the mean of these losses.

    Args:
    obs_prediction (Dict[str, torch.Tensor]): The predicted observation as a dictionary of tensors.
    next_obs (Dict[str, torch.Tensor]): The actual next observation as a dictionary of tensors.

    Returns:
    torch.Tensor: The mean loss across all components.
    """
    
    if obs_prediction is None or next_obs is None:
        return 0 
    if len(obs_prediction) == 1:
        key = list(obs_prediction.keys())[0]
        if key in next_obs:
            squared_error = torch.pow(obs_prediction[key] - next_obs[key], 2)
            return torch.mean(squared_error, dim=tuple(range(1, squared_error.dim())))
        else:
            print(f"Warning: Key '{key}' found in obs_prediction but not in next_obs")
            return 0
        
    
    losses = []
    for key in obs_prediction:
        if key in next_obs:
            squared_error = torch.pow(obs_prediction[key] - next_obs[key], 2)
            losses.append(torch.mean(squared_error, dim=tuple(range(1, squared_error.dim()))))
        else:
            print(f"Warning: Key '{key}' found in obs_prediction but not in next_obs")

    if not losses:
        raise ValueError("No corresponding keys found in obs_prediction and next_obs")

    
    # Stack the tensors along a new dimension (dim=0)
    stacked_tensor = torch.stack(losses, dim=0)

    # Calculate the mean along dimension 0 to obtain a tensor of shape (a,)
    mean_tensor = torch.mean(stacked_tensor, dim=0)
    return mean_tensor