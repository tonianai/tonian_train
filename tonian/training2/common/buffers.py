from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple

from tonian.common.spaces import MultiSpace

from gym import spaces

import numpy as np
import torch

"""
The Buffer structure was largely insipred by stable-baselines-3 implementation
But written in such a way, that the data does not have to leave the gpu memory 
"""



class BaseBuffer(ABC):

    def __init__(self) -> None:
        super().__init__()
    

class DictExperienceBufferSamples(NamedTuple):
    critic_obs: Dict[Union[int, str], torch.Tensor]
    actor_obs: Dict[Union[int, str], torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class DictExperienceBuffer(BaseBuffer):
    """Dict Experience buffer used in on policy algorithms
     Extends the ExperienceBuffer to use dictionary observations
     
    It corresponds to ``horizon_length`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    
    The buffer saves tensors, in order to keep them on the Gpu and reduce transfers between gpu and cpu 
    The grad property will not be touched

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

        Args:
            horizon_length (int): Number of steps saved in the buffer 
            critic_obs_space (MultiSpace): Observation Space used by the critic, also contains the commands if they exist
            actor_obs_space (MultiSpace): Observation Spaces used by the actor, also contains the commands if they exist
            action_space (spaces.Space): Action Space
            store_device (Union[str, torch.device], optional): [The Device on which the values will be stored. Defaults to "cuda:0".
                (If possible, let all tensors stay on the gpu, to minimize cpu usage)
            out_device (Union[str, torch.devie]) The device on which the tensors will be outputted from (This is preferably the same as the store_device )
            n_envs (int, optional): Number of paralell environments. Defaults to 1.
            n_actors (int, optional): Number of actors per environment
    """
    
    def __init__(self,
                 horizon_length: int,
                 critic_obs_space: MultiSpace,
                 actor_obs_space: MultiSpace,
                 action_space: spaces.Space,
                 store_device: Union[str, torch.device] = "cuda:0",
                 out_device: Union[str, torch.device] = "cuda:0",
                 n_envs: int = 1,
                 n_actors: int = 1
                 ) -> None:
        super().__init__()
        
        self.horizon_length = horizon_length
        self.critic_obs_space = critic_obs_space
        self.actor_obs_space = actor_obs_space
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        
        assert self.action_size, "Action size must not be zero"
            
        self.store_device = store_device
        self.out_device = out_device
        
        # save the dict shape of the actor and the critic
        self.critic_obs_dict_shape = critic_obs_space.dict_shape
        self.actor_obs_dict_shape = actor_obs_space.dict_shape
        
        # current fill position of the buffer
        self.pos = 0
        self.full = False
        self.n_envs = n_envs
        self.n_actors = n_actors
        
        self.n_actors_per_step = n_envs * n_actors
        
        self.reset()
        
        
        
    def reset(self) -> None:
        """
        Reset the buffer
        """
        self.pos = 0
        self.full = False
        
        # store everything in torch tensors on the gpu
        self.actions = torch.zeros((self.horizon_length, self.n_actors_per_step, self.action_size), dtype=torch.float32, device=self.store_device)
        self.rewards = torch.zeros((self.horizon_length, self.n_actors_per_step), dtype=torch.float32, device=self.store_device)
        self.returns = torch.zeros((self.horizon_length, self.n_actors_per_step), dtype=torch.float32, device=self.store_device)
        self.dones = torch.zeros((self.horizon_length, self.n_actors_per_step), dtype=torch.int8, device=self.store_device)
        self.values = torch.zeros((self.horizon_length, self.n_actors_per_step), dtype=torch.float32, device=self.store_device)
        self.neglogpacs = torch.zeros((self.horizon_length, self.n_actors_per_step), dtype=torch.float32, device=self.store_device)
        
        # the mean of the action distributions
        self.action_means = torch.zeros((self.horizon_length, self.n_actors_per_step, self.action_size), dtype=torch.float32)
        # the std(sigma) of the action distributions   
        self.action_std = torch.zeros((self.horizon_length, self.n_actors_per_step, self.action_size), dtype= torch.float32)
        
        
        # critic obs and actor obs must be dicts, because this enables multispace environments
        self.critic_obs = {}
        self.actor_obs = {}
        
        for key, obs_shape in self.actor_obs_dict_shape.items():
            self.actor_obs[key] = torch.zeros((self.horizon_length, self.n_envs) + obs_shape, dtype=torch.float32, device= self.store_device)
        
        for key, obs_shape in self.critic_obs_dict_shape.items():
            self.critic_obs[key] = torch.zeros((self.horizon_length, self.n_envs) + obs_shape, dtype=torch.float32, device= self.store_device)
        
    def add(
        self, 
        actor_obs: Dict[str, torch.Tensor],
        critic_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        neglogpacs: torch.Tensor,
        action_means: torch.Tensor,
        action_stds: torch.Tensor
    ) -> None:
        """
        :param critic_obs: Observation of the critic, also contains commands, if available
        :param actor_obs: Observations of the actor, also contains commands, if available
        :param reward: Rewards gotten by the env
        :param dones: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param neglogpacs: neg log probability of the action
            following the current policy.
        :param action_means: the mean of the gaussian distribution the action was sampled from
        :param action_stds: the stds of the gaussian distribution the action was sampled from 
        """
        
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
            
        for key in self.actor_obs:   
            self.actor_obs[key][self.pos] = actor_obs[key].detach().to(self.store_device)
  
            
        for key in self.critic_obs:
            self.critic_obs[key][self.pos] = critic_obs[key].detach().to(self.store_device)
         
        
        self.actions[self.pos] = actions.detach().to(self.store_device)
        self.rewards[self.pos] = rewards.detach().to(self.store_device)
        self.dones[self.pos] = dones.detach().to(self.store_device)
        self.values[self.pos] = values.detach().squeeze().to(self.store_device)
        self.neglogpacs[self.pos] = neglogpacs.detach().squeeze().to(self.store_device)
        self.action_means[self.pos] = action_means.detach().squeeze().to(self.store_device)
        self.action_std[self.pos] = action_stds.detach().squeeze().to(self.store_device)
    
        
        self.pos += 1
        
        if self.pos == self.horizon_length:
            self.full = True
            
        
    def size(self) -> int:
        """
        Returns:
            int: current size of the buffer
        """
        if self.full:
            return self.horizon_length
        return self.pos
     
    
 
 
        



        
        
        
        
    
    