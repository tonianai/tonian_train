from typing import Dict, Union, List, Tuple, Type, Optional

import torch 
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions import MultivariateNormal

from collections import deque

from elysium.algorithms.base_algorithm import BaseAlgorithm
from elysium.common.buffers import DictRolloutBuffer
from elysium.common.schedule import Schedule
from elysium.tasks.base.vec_task import VecTask

from elysium.algorithms.policies import ActorCriticPolicy

import numpy as np

 

import time



class PPO(BaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, policy :ActorCriticPolicy, device: Union[str, torch.device], log_compute_rewards: bool = True) -> None:
        super().__init__(env, config, device)
        self._fetch_config_params(config)
        
        
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
        
        # the step when the last save was made
        self.last_save = 0
        
        self.buffer_size = self.env.num_envs * self.n_steps
        
        self._last_obs = None # Type Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]] # first dirct is critic obs last dice is actor obs
        
        assert self.batch_size > 1, "Batch size must be bigger than one"
        
        assert self.buffer_size > 1, "Buffer size must be bigger than one"
        
        assert self.buffer_size % self.batch_size == 0, "the buffer size must be a multiple of the batch size"
        
        self._setup_model()
        
        
        
    def _fetch_config_params(self, config):
        """Fetćh hyperparameters from the configuration dict and set them to member variables
        
        Args:
            config ([type]): [description]
        """
        
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.n_steps = config['n_steps']
        self.target_kl = config['target_kl']
        self.gae_lambda = config['gae_lamda']
        self.eps_clip = config['eps_clip']
        self.value_f_coef = config['value_f_coef']
        self.entropy_coef = config['entropy_coef']
         
        
        
    def _setup_model(self) -> None:
        """Setup the model
                - setup schedules
                - initialize buffer
        """
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
        
        
    def _setup_learn(
        self, 
        total_timesteps: int, 
        reset_num_timesteps: bool = True):
        self.start_time = time.time()
        
        if reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)
            
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps
        
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        
        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = torch.ones((self.env.num_envs,), dtype=torch.int8, device= self.device)
            
    def learn(
        self, 
        total_timesteps: int,
        reset_num_timesteps: bool = True
    )-> None:
        
        iteration = 0
        
        self._setup_learn(total_timesteps, reset_num_timesteps)
        
        total_timesteps = self._total_timesteps
                
        while self.num_timesteps < total_timesteps:
            
            continue_training = self.collect_rollouts(n_rollout_steps=self.n_steps)
            
            iteration += 1
            
            self.train()
            
    
    def collect_rollouts(self, 
                         n_rollout_steps: int) -> bool:
        """Collects experiences using the current policy and fills the self.rollout_buffer

        Args:
            n_rollout_steps (int): Number of experiences to collect per environment
        
        Return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        print("collect rollouts")
        
        assert self._last_obs is not None, "No previous obs was provided"
        self.policy.train(False)
    
    
        n_steps = 0 
        self.rollout_buffer.reset()
        
        # cumulative sum of  episode rewards within rollout ()
        sum_ep_reward = 0
        
        # cumulative sum of the amount of completed episodes
        n_completed_episodes = 0
        
        # cumulative sum of all the steps taken in all the episodes
        sum_steps_per_episode = 0
        
        while n_steps < n_rollout_steps:
            
            with torch.no_grad():
                actions, values, log_probs = self.policy.forward(self._last_obs[0], self._last_obs[1])
                
            
            # clamp the action space using pytorch
            clipped_actions = torch.clamp(actions, self.action_low_torch, self.action_high_torch)
            
            new_obs, rewards, dones, info = self.env.step(clipped_actions)
             
            # add all the episodes that were completed whitin the last time step to the counter
            n_completed_episodes +=  torch.sum(dones).item()
            
            # sum of all rewards of all completed episodes
            sum_ep_reward += torch.sum(info["episode_reward"]).item()
            
            # sum all the steps of all completed episodes
            sum_steps_per_episode  += torch.sum(info["episode_steps"]).item()
            
            
            self.num_timesteps += self.env.num_envs
            
            n_steps += 1
            
            
            # refer to stable-baselines on_policy_alorgithm.py line 196 
            
            
            self.rollout_buffer.add(self._last_obs[0], self._last_obs[1], actions, rewards, self._last_episode_starts, values, log_probs)
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        

        # log the rollout information 
        self.logger.log("run/episode_rewards", sum_ep_reward / n_completed_episodes, self.num_timesteps)
        self.logger.log("run/steps_per_episode", sum_steps_per_episode / n_completed_episodes, self.num_timesteps)
        
        with torch.no_grad():
            # compute the value for the last timestep
            values = self.policy.predict_values(new_obs[1])

            
        
            
        self.rollout_buffer.compute_returns_and_advantages(values.squeeze(), dones)
        
        return True        
    
    def train(self)-> None:
        """Update the policy using the rollout buffer
        """        
        # switch to train mode
        self.policy.train(True)
        # update schedules (like lr)
        print("Train")
        
        self._update_schedules()        
        
        
        entropy_losses = []
        pg_losses = []
        value_losses = []
        
        clip_fractions = []
        
        continue_training = True
        
        
        clip_range = self.eps_clip
        # todo introduce schedule
        
        for epoch in range(self.n_epochs):
            # train for each epoch
            
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.actor_obs, rollout_data.critic_obs, actions)
                
                values = values.flatten()
                
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                
                # todo: maybe introduce value function clipping
                
                # Value loss using the TD(gae_lambda
                
                value_loss = F.mse_loss(rollout_data.returns.squeeze(), values)
                
                value_losses.append(value_loss.item())
                
                
                entropy_loss = torch.mean(entropy)
                
                entropy_losses.append(entropy_loss.item())
                
                
                loss = policy_loss + self.entropy_coef * entropy_loss + self.value_f_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                    
                # Optimization Step
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
                
        
        self.logger.log("train/entropy_loss", np.mean(entropy_losses), self.num_timesteps)       
        self.logger.log("train/value_loss", np.mean(value_losses), self.num_timesteps)
        self.logger.log("train/action_std", self.policy.log_std.exp().mean().item(), self.num_timesteps)
        self.logger.log("train/loss", loss.item(), self.num_timesteps)
        self.logger.log("train/clip_fraction", np.mean(clip_fractions), self.num_timesteps)
        self.logger.log("train/policy_gradient_loss", np.mean(pg_losses), self.num_timesteps)
        self.logger.log("train/approx_kl_div", np.mean(approx_kl_div), self.num_timesteps)
        
        # TODO: add explained variance
        if self.num_timesteps - self.last_save > 1e6 or self.last_save == 0:
            self.save()
            self.last_save = self.num_timesteps
        
        
        
    pass

    def _update_schedules(self):
        # todo add lr schedule and not a fixed rate
        # change 
        
        pass
     
    
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
    
    