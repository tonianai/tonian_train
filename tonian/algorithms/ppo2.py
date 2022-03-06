from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
import torch
from gym import spaces
from torch.nn import functional as F

from tonian.algorithms.base_algorithm import BaseAlgorithm
from tonian.policies.policies import ActorCriticPolicy
from tonian.tasks.base.vec_task import VecTask
from tonian.common.buffers import DictRolloutBuffer


class PPO(BaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, policy: ActorCriticPolicy ,device: Union[str, torch.device]) -> None:
        super().__init__(env, config, device)
        
        # fetch the arguments from the config Dict
        self.gamma = config['gamma']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.n_steps = config['n_steps']
        if 'tarket_kl' in config:
            self.target_kl = config['target_kl']
        else:
            self.target_kl = None
        self.gae_lambda = config['gae_lamda']
        self.eps_clip = config['eps_clip']
        self.value_f_coef = config['value_f_coef']
        self.entropy_coef = config['entropy_coef']
        
        
        # set the action and obervation space to member variables
        self.critic_obs_spaces = env.critic_observation_spaces
        self.actor_obs_spaces = env.actor_observation_spaces
        self.action_space = env.action_space
        
        # the torch tensor for the min action values
        self.action_low_torch = torch.as_tensor(self.action_space.low, device=self.device)
        # the torch tensor for the max action values
        self.action_high_torch = torch.as_tensor(self.action_space.high, device= self.device)
        
        self.actor_obs_shapes = env.actor_observation_spaces.shape
        self.critic_obs_shapes = env.critic_observation_spaces.shape
        
        # set the amount of envs as a member variable
        self.n_envs = env.num_envs
        
        self.policy = policy.to(self.device)
        
        self.lr = self.policy.lr_schedule
        
        # the step when the last save was made
        self.last_save = 0
        
        self.buffer_size = self.env.num_envs * self.n_steps
        
        self._last_obs = None # Type Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]] # first dirct is critic obs last dice is actor obs
        self._last_episode_starts = torch.ones((self.n_envs, ), dtype=torch.int8, device= self.device)
        
        assert self.batch_size > 1, "Batch size must be bigger than one"
        
        assert self.buffer_size > 1, "Buffer size must be bigger than one"
        
        assert self.buffer_size % self.batch_size == 0, "the buffer size must be a multiple of the batch size"
        

        self.rollout_buffer = DictRolloutBuffer(
            self.n_steps,
            self.critic_obs_spaces,
            self.actor_obs_spaces,
            self.action_space,
            self.device,
            gamma= self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )
    
    def collect_rollouts(self, n_rollout_steps: int):
        """
        Collect rollouts in the environment and save them to the rollout buffer

        Args:
            n_rollout_steps (int): Number of steps taken
        """
        print("collect rollouts")
        
        i_step = 0
        self.rollout_buffer.reset()
        
        if self._last_obs is None:
            self._last_obs = self.env.reset()
        
        while i_step < n_rollout_steps:
            
            with torch.no_grad():
                action, value, log_prob = self.policy.forward(actor_obs=self._last_obs[0], critic_obs=self._last_obs[1])
                
            new_obs, rewards, dones, info = self.env.step(actions= action)
            
            
            self.rollout_buffer.add(
                actor_obs=self._last_obs[0],
                critic_obs=self._last_obs[1],
                action = action,
                reward = rewards,
                is_epidsode_start= self._last_episode_starts,
                value = value,
                log_prob=log_prob
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            i_step += 1
        
        
        with torch.no_grad():
            # compute the value for the last timestep
            values = self.policy.predict_values(new_obs[1])

        self.rollout_buffer.compute_returns_and_advantages(values.squeeze(), dones)
    
             
    def learn(self, total_timesteps: int) -> None:
        """ Jump between rollout and training and learn a better policy using ppo

        Args:
            total_timesteps (int): 
        """
        
        n_steps_trained = 0
        
        while n_steps_trained < total_timesteps:
            
            self.collect_rollouts(n_rollout_steps=self.n_steps)
             
            self.train()
            
            n_steps_trained += self.n_steps
            
        
            
    def train(self) -> None:
        print("Train") 
        
        
        
        # Do a complete pass on the rollout buffer
        for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                actions = rollout_data.actions
                
                
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.actor_obs, rollout_data.critic_obs, actions)
                 
                
                print('new Log prob')
                print(log_prob)
                print('old log prob')
                print(rollout_data.old_log_prob)
  
        
        
    def save(self, path: Optional[str] = None):
        
        if path is None:
            path = self.run_folder_name + "/saves/" + str(self.num_timesteps) + ".pth"
        
        self.policy.save(path)


    def load(self, path: str):
        """Load from a given checkpoint
        Args:
            path (str): [description]
        """
        self.policy.load(path)
        
        
 
        
        
    
