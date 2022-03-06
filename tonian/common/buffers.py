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
        
    @staticmethod
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
        

class DictRolloutBufferSamples(NamedTuple):
    critic_obs: Dict[Union[int, str], torch.Tensor]
    actor_obs: Dict[Union[int, str], torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class DictRolloutBuffer(BaseBuffer):
    """Dict Rollout buffer used in on policy algorithms
     Extends the RolloutBuffer to use dictionary observations
     
    It corresponds to ``buffer_size`` transitions collected
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
            buffer_size (int): Max number of element in the buffer
            critic_obs_space (MultiSpace): Observation Space used by the critic
            actor_obs_space (MultiSpace): Observation Spaces used by the actor
            action_space (spaces.Space): Actoion PSace
            device (Union[str, torch.device], optional): [The Device on which the values will be stored. Defaults to "cuda:0".
                (If possible, let all tensors stay on the gpu, to minimize cpu usage)
                ((Might implement intelligent device switching in the future))
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                Equivalent to classic advantage when set to 1.
            gamma: Discount factor, Defaults to 0.99
            n_envs (int, optional): Number of paralell environments. Defaults to 1.
    """
    
    def __init__(self,
                 buffer_size: int,
                 critic_obs_space: MultiSpace,
                 actor_obs_space: MultiSpace,
                 action_space: spaces.Space,
                 device: Union[str, torch.device] = "cuda:0",
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1
                 ) -> None:
        super().__init__()
        
        self.buffer_size = buffer_size
        self.critic_obs_space = critic_obs_space
        self.actor_obs_space = actor_obs_space
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        self.generator_ready = False

    
    
        assert self.action_size, "Action size must not be zero"
            
        self.device = device
        
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        
        # save the dict shape of the actor and the critic
        self.critic_obs_dict_shape = critic_obs_space.dict_shape
        self.actor_obs_dict_shape = actor_obs_space.dict_shape
        
        # current fill position of the buffer
        self.pos = 0
        self.full = False
        self.n_envs = n_envs
        
        self.reset()
        
        
        
    def reset(self) -> None:
        """
        Reset the buffer
        """
        self.pos = 0
        self.full = False
        
        # store everything in torch tensors on the gpu
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_size), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.is_epidsode_start = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.int8, device=self.device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        
        self.generator_ready = False
        
        # critic obs and actor obs must be dicts, because this enables multispace environments
        self.critic_obs = {}
        self.actor_obs = {}
        
        for key, obs_shape in self.actor_obs_dict_shape.items():
            self.actor_obs[key] = torch.zeros((self.buffer_size, self.n_envs) + obs_shape, dtype=torch.float32, device= self.device)
        
        for key, obs_shape in self.critic_obs_dict_shape.items():
            self.critic_obs[key] = torch.zeros((self.buffer_size, self.n_envs) + obs_shape, dtype=torch.float32, device= self.device)
        
    def add(
        self, 
        actor_obs: Dict[str, torch.Tensor],
        critic_obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        is_epidsode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ) -> None:
        """
        :param critic_obs: Observation 
        :param actor_obs: Action
        :param reward:
        :param episode_start: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        
        """
        print("OBS Shape")
        print(actor_obs['linear'].shape)
        
        print("OBS critic Shape")
        print(critic_obs['linear'].shape)
            
        print("Actions Shape")
        print(action.shape)
    
        print("Reward SHape")
        print(reward.shape)
        
        print("Value Shape")
        print(value.shape)"""
        
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
            
            
        for key in self.actor_obs:          
            self.actor_obs[key][self.pos] = actor_obs[key].detach().clone()
  
            
        for key in self.critic_obs:
            self.critic_obs[key][self.pos] = critic_obs[key].detach().clone()
        
        
        self.actions[self.pos] = action.detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.is_epidsode_start[self.pos] = is_epidsode_start.detach().clone()
        self.values[self.pos] = value.detach().clone().squeeze()
        self.log_probs[self.pos] = log_prob.detach().clone().squeeze()
        
        
        
        
        self.pos += 1
        
        if self.pos == self.buffer_size:
            self.full = True
            
    def get(self, batch_size: Optional[int] = None) -> Generator[DictRolloutBufferSamples, None, None]:
        """This is a generator function, that returns a batch_sizes worth of buffer samples, but only for a full buffer

        Args:
            batch_size (Optional[int], optional): The batch size of the returned batch, it it is none, the whole buffer will be returned. Defaults to None.

        Yields:
            Generator[DictRolloutBufferSamples, None, None]: 
        """
        
        assert self.full, "The buffer must be full, before retreiving data from it"
        
        
        # prepare the data by flattening the n_envs 
        # use view, because we don't want contigous tensors for performance reasons (no flatten or reshape)
        
        if not self.generator_ready: 
            
            self.actions = self.swap_and_flatten(self.actions)
            self.rewards = self.swap_and_flatten(self.rewards)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            
            self.generator_ready = True
        
            for key, obs in self.actor_obs.items():
                self.actor_obs[key] = self.swap_and_flatten(obs)
         
            for key in self.critic_obs:
                self.critic_obs[key] = self.swap_and_flatten(obs)
            
        
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
            
        
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        #indices = np.arange(stop= self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
        
    def _get_samples(self, batch_inds: np.ndarray) -> DictRolloutBufferSamples:
        
        return DictRolloutBufferSamples (
            critic_obs={key: obs[batch_inds] for (key, obs) in self.critic_obs.items()},
            actor_obs={key: obs[batch_inds] for (key, obs) in self.actor_obs.items()},
            actions= self.actions[batch_inds],
            old_values= self.values[batch_inds].squeeze(),
            old_log_prob= self.log_probs[batch_inds].squeeze(),
            advantages= self.advantages[batch_inds].squeeze(),
            returns= self.returns[batch_inds].squeeze()
        )
         
    def size(self) -> int:
        """
        Returns:
            int: current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos
     
    def compute_returns_and_advantages(self, last_values: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env) shape(num_envs)
        :param dones: if the last step was a terminal step (one bool for each env). shape(num_envs)
        """
        
        last_gae_lam = 0
        
        #print(last_values.shape)
        last_values = last_values.clone().flatten()
        #print(last_values.shape)
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.is_epidsode_start[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        
    
 
 
        



        
        
        
        
    
    