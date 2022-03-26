from typing import Callable, Dict, Union, List, Tuple, Type, Optional

import torch 
import torch.nn as nn
from torch.nn import functional as F
  
from collections import deque

from tonian.algorithms.base_algorithm import BaseAlgorithm
from tonian.common.buffers import DictRolloutBuffer
from tonian.common.schedule import Schedule, schedule_or_callable
from tonian.tasks.base.vec_task import VecTask
from tonian.common.utils.utils import set_random_seed
from tonian.common.logger import BaseLogger

from tonian.policies.policies import ActorCriticPolicy

import numpy as np
import yaml, os, time



class PPO(BaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, policy :ActorCriticPolicy, device: Union[str, torch.device], logger: BaseLogger) -> None:
        super().__init__(env, config, device, logger)
        self._fetch_config_params()
         
        
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
        
        self.highest_avg_reward = -1e9
        
        self.current_avg_reward = -1e9 # the reward obtained most recently
        
        assert self.batch_size > 1, "Batch size must be bigger than one"
        
        assert self.buffer_size > 1, "Buffer size must be bigger than one"
        
        assert self.buffer_size % self.batch_size == 0, "the buffer size must be a multiple of the batch size"
        
        self._setup_model()
        
        
        
        
    def _fetch_config_params(self):
        """Fetćh hyperparameters from the configuration dict and set them to member variables
         
        """
        
        self.gamma = self.config['gamma']
        self.n_epochs = self.config['n_epochs']
        self.batch_size = self.config['batch_size']
        self.n_steps = self.config['n_steps']
        if 'tarket_kl' in self.config:
            self.target_kl = self.config['target_kl']
        else:
            self.target_kl = None
        self.gae_lambda = self.config['gae_lamda']
        self.eps_clip = self.config['eps_clip']
        self.value_f_coef = self.config['value_f_coef']
        self.entropy_coef = self.config['entropy_coef']
        self.save_freq = float(self.config['save_freq'])
        
         
        self.max_grad_norm = None
        if 'max_grad_norm' in self.config:
            self.max_grad_norm = self.config['max_grad_norm']
         
        
        
    def _setup_model(self) -> None:
        """Setup the model
                - setup schedules
                - set seed
                - initialize buffer
        """
        self.rollout_buffer = DictRolloutBuffer(
            self.n_steps,
            self.critic_obs_spaces,
            self.actor_obs_spaces,
            self.action_space,
            store_device= self.device,
            out_device= self.device,
            gamma= self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )
        
        if "seed" in self.config:
            # set the seed of random, torch, the env and numpy
            seed = self.config["seed"]
            
            print(f"Seed: {seed}")
            set_random_seed(seed=seed, using_cuda=  "cuda" in self.device)
            self.action_space.seed(seed)
            
        
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
        """Alternate between rollout and model optimization 

        Args:
            total_timesteps (int): The amount of timesteps all robots combined should take
            reset_num_timesteps (bool, optional): 
        """
        iteration = 0
        
        self._setup_learn(total_timesteps, reset_num_timesteps)
        
        total_timesteps = self._total_timesteps
                
        while self.num_timesteps < total_timesteps:
            
            start_time = time.time()
            
                
            self.collect_rollouts(n_rollout_steps=self.n_steps)
            
            end_rollout_time = time.time()
            iteration += 1
        
            self.train()
            
            end_time = time.time()
            
            print(" Run: {}    |     Iteration: {}     |    Steps Trained: {:.3e}     |     Steps per Second: {:.0f}     |     Time Spend on Rollout: {:.2%}".format(self.logger.identifier ,iteration, self.num_timesteps,(self.n_steps * self.n_envs)/ (end_time - start_time), ( end_rollout_time - start_time) / (end_time - start_time )))
            
    
    def collect_rollouts(self, 
                         n_rollout_steps: int) -> bool:
        """Collects experiences using the current policy and fills the self.rollout_buffer

        Args:
            n_rollout_steps (int): Number of experiences to collect per environment
        
        Return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """ 
        
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
                
                actor_obs = { key : tensor_obs.detach().clone() for (key, tensor_obs) in self._last_obs[0].items()}
                
                critic_obs = { key : tensor_obs.detach().clone() for (key, tensor_obs) in self._last_obs[1].items()}
                 
            
            # clamp the action space using pytorch
            #clipped_actions = torch.clamp(actions, self.action_low_torch, self.action_high_torch)
            
            new_obs, rewards, dones, info = self.env.step(actions)
             
            # add all the episodes that were completed whitin the last time step to the counter
            n_completed_episodes +=  torch.sum(dones).item()
            
            # sum of all rewards of all completed episodes
            sum_ep_reward += torch.sum(info["episode_reward"]).item()
            
            # sum all the steps of all completed episodes
            sum_steps_per_episode  += torch.sum(info["episode_steps"]).item()
            
            
            self.num_timesteps += self.env.num_envs
            
            n_steps += 1
            
            
            # refer to stable-baselines on_policy_alorgithm.py line 196 
            
            
            self.rollout_buffer.add(
                actor_obs = actor_obs, 
                critic_obs = critic_obs, 
                action = actions, 
                reward = rewards, 
                is_epidsode_start= self._last_episode_starts, 
                value = values, 
                log_prob =log_probs)
            
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        

        # log the rollout information 
        self.logger.log("run/episode_rewards", sum_ep_reward / n_completed_episodes, self.num_timesteps)
        self.logger.log("run/steps_per_episode", sum_steps_per_episode / n_completed_episodes, self.num_timesteps)
        
        with torch.no_grad():
            # compute the value for the last timestep
            values = self.policy.predict_values(new_obs[1])

        self.current_avg_reward = sum_ep_reward / n_completed_episodes
        
        self.rollout_buffer.compute_returns_and_advantages(values.squeeze(), dones)
        
        return True        
    
    def train(self)-> None:
        """Update the policy using the rollout buffer
        """        
        # switch to train mode
        self.policy.train(True)
        # update schedules (like lr) 
        
        self._update_schedules()        
        
        
        entropy_losses = []
        pg_losses = []
        value_losses = []
        
        clip_fractions = []
        
        
        clip_range = self.eps_clip
        # todo introduce schedule
        
        for _ in range(self.n_epochs):
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
                # Clip grad norm
                
                if self.max_grad_norm is not None: 
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                
        
        self.logger.log("train/entropy_loss", np.mean(entropy_losses), self.num_timesteps)       
        self.logger.log("train/value_loss", np.mean(value_losses), self.num_timesteps)
        self.logger.log("train/action_std", self.policy.log_std.exp().mean().item(), self.num_timesteps)
        self.logger.log("train/loss", loss.item(), self.num_timesteps)
        self.logger.log("train/clip_fraction", np.mean(clip_fractions), self.num_timesteps)
        self.logger.log("train/policy_gradient_loss", np.mean(pg_losses), self.num_timesteps)
        self.logger.log("train/approx_kl_div", np.mean(approx_kl_div), self.num_timesteps)
        
        
        
        if self.num_timesteps - self.last_save > 1e7 or self.last_save == 0:
            self.save()
            self.last_save = self.num_timesteps

            if self.highest_avg_reward < self.current_avg_reward:
                self.highest_avg_reward = self.current_avg_reward
                # save the run, that performed the best as such
                self.save(save_as_best=True)

    def _update_schedules(self):
        # todo add lr schedule and not a fixed rate
        # change 
        
        pass
     
    
    def save(self, path: Optional[str] = None, save_as_best: bool = False):
        """Save the current policy in the run folder

        Args:
            path (Optional[str], optional): path at which the policy should be saved. If none use the default run folder. Defaults to None.
            save_as_best (bool, optional): Determines whether this run should be saved as the best run. And therefore override the previous best run. Defaults to False.
        """
        
        if not save_as_best:
            file_name = str(self.num_timesteps) + ".pth"
        else:
            file_name = "best.pth"
            
        
        if path is None:
            path = os.path.join( self.logger.save_folder , file_name )
        
        self.policy.save(path)


    def load(self, path: str):
        """Load from a given checkpoint
        Args:
            path (str): [description]
        """
        self.policy.load(path)
    
    def _get_standard_config(self) -> Dict:
        """Get the standard configuration for the ppo algorithm

        Returns:
            Dict: _description_
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'ppo_config.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
    