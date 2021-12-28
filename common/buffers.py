from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple


import numpy as np
import torch



class RolloutBufferSamples(NamedTuple):
    obss: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class BaseBuffer(ABC):

    
    def __init__(self, 
                 buffer_size:int,
                 linear_obs_size: int,
                 visual_obs_shape: Tuple[int, int, int],
                 command_size: int,
                 action_size: int,
                 device: Union[torch.device, str] = "cpu",
                 n_agents: int = 1
                 ) -> None:
        """ Base class that represents a buffer (rollout or replay)

        Args:
            buffer_size (int): Max number of element in the buffer
            linear_obs_size (int): size of the 1d observation vector
            visual_obs_shape (Union): shape of the linear observation (x, y, channels) (0,0,0) if there is no visual obs
            command_size (int): size of the command to execute
            action_size (int): size of the action
            device (Union[torch.device, str], optional): [description]. Defaults to "cpu".
            n_agents (int, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.linear_obs_size = linear_obs_size
        self.visual_obs_shape = visual_obs_shape
        self.command_size = command_size
        self.action_size = action_size
        self.device = device
        self.full = False
        self.pos = 0 # position the buffer is currently filled to
        self.n_agents = n_agents
        
    def size(self) -> int:
        return self.pos
    
    
    def reset(self) -> None:
        """
        Reset(empty) the Buffer
        """
        self.pos = 0
        self.full = False
        
    def sample(self, batch_size: int):
        """
        Args:
            batch_size (int): number of elements to sample
        """
        upper_bound = self.buffer_size if self.full else self.pos
            
    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray= None
    ) -> RolloutBufferSamples:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()
    
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
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
    
    
class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    
    (This was stolen from stable-baselines3)
    """
    
    def __init__(self, 
                 buffer_size: int,
                 linear_obs_size: int,
                 visual_obs_shape: Tuple[int, int, int], 
                 command_size: int, 
                 action_size: int, 
                 device: Union[torch.device, str] = "cpu", 
                 gae_lamda: float = 1,
                 gamma: float = 0.99,
                 n_agents: int = 1) -> None:
        """

        Args:
            buffer_size (int): Max Number of Elements in the Buffer
            linear_obs_size (int): Size of the 1d observation space
            visual_obs_shape (Tuple): shape of the linear observation (x, y, channels) (0,0,0) if there is no visual obs
            command_size (int): Size of the command received
            action_size (int): Size of the action of the robot
            device (Union[torch.device, str], optional): [description]. Defaults to "cpu".
            gae_lamda (float, optional): Factor for trade-off of bias vs variance for Generalized Advantage Estimator
           Equivalent to classic advantage when set to 1. Defaults to 1.
            gamma (float, optional): Discount Fctort. Defaults to 0.99.
            n_agents (int, optional): Number of parallel agents. Defaults to 1.
        """
        super().__init__(buffer_size, linear_obs_size, visual_obs_shape, command_size, action_size, device=device, n_agents=n_agents)
        self.gae_lamda = gae_lamda
        self.gamma = gamma
        self.linear_observations, self.visual_observations, self.commands ,self.actions, self.rewards, self.advantages = None, None, None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()
        
        
    def reset(self) -> None:
        
        self.linear_observations = np.zeros((self.buffer_size, self.n_agents, self.linear_obs_size), dtype=np.float32)
        self.visual_observations = np.zeros((self.buffer_size, self.n_agents) + self.visual_obs_shape )
        self.commands = np.zeros((self.buffer_size, self.n_agents, self.command_size), dtype= np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_agents, self.action_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.generator_ready = False
        super().reset()
        
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor ) -> None:
        
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

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        
        (This also was stolen from stable-baselines 3)
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        
    def add(
        self,
        obs:np.ndarray,
        visual_obs: np.ndarray,
        command: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ) -> None:
        """Add a new experience to the buffer

        Args:
            obs (np.ndarray): linear observation shape: (num_agents, obs_size)
            visual_obs (np.ndarray): visual observation
            commands (np.ndarray): [description]
            action (np.ndarray): [description]
            reward (np.ndarray): [description]
            episode_start (np.ndarray): [description]
            value (torch.Tensor): [description]
            log_prob (torch.Tensor): [description]
        """
        
        
        self.linear_observations[self.pos] = np.array(obs).copy()
        self.visual_observations[self.pos] = np.array(visual_obs).copy()
        self.commands[self.pos] = np.array(command).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        
        if self.pos == self.buffer_size:
            self.full = True
            
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "The Buffer should be full in order to sample it"
        
        indices = np.random.permutation(self.buffer_size * self.n_agents)
        
        # Prepare the data
        if not self.generator_ready:
            
            _tensor_names = [
                "linear_observations",
                "visual_observations",
                "commands",
                "actions",
                "advantages",
                "returns"
            ]
            
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
            
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_agents
        
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

            
    def _get_samples(self, batch_inds: np.ndarray = None) -> RolloutBufferSamples:
        data = (
            self.linear_observations[batch_inds],
            self.visual_observations[batch_inds],
            self.commands[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
        
        
        
        
