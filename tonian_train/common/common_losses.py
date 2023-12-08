from torch import nn
import torch
import torch.nn.functional as F
from typing import Dict, Optional

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
    losses = []
    for key in obs_prediction:
        if key in next_obs:
            loss = F.mse_loss(obs_prediction[key], next_obs[key])
            losses.append(loss)
        else:
            print(f"Warning: Key '{key}' found in obs_prediction but not in next_obs")

    if not losses:
        raise ValueError("No corresponding keys found in obs_prediction and next_obs")

    total_loss = torch.mean(torch.stack(losses))
    return total_loss