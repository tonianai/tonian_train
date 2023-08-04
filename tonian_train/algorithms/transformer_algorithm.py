
from typing import Dict, Tuple, Any, Union, Optional
from abc import ABC, abstractmethod
from collections import deque
import numpy as np

from tonian_train.tasks import VecTask
from tonian_train.common.logger import BaseLogger, TensorboardLogger
from tonian_train.common.spaces import MultiSpace
from tonian_train.common.schedulers import AdaptiveScheduler, LinearScheduler, IdentityScheduler
from tonian_train.policies import TransformerPolicy
from tonian_train.common.helpers import DefaultRewardsShaper
from tonian_train.common.running_mean_std import RunningMeanStd, RunningMeanStdObs
from tonian_train.common.buffers import DictExperienceBuffer
from tonian_train.common.common_losses import critic_loss, actor_loss
from tonian_train.common.dataset import PPODataset
from tonian_train.common.utils import join_configs

import torch.nn as nn
import torch, gym, os, yaml, time

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl
    
    
class TransformerBuffer():
    
    pass



class SequenceBuffer():
    
    def __init__(self,
                 n_envs: int,  
                 horizon_length: int,
                 obs_space: MultiSpace,
                 action_space: gym.spaces.Space,
                 store_device: Union[str, torch.device] = "cuda:0",
                 out_device: Union[str, torch.device] = "cuda:0",
                 n_values: int = 1) -> None:
        """Buffer, that stores the observations and the outputs for the transformer sequence length
        
        Args:
            sequence_length (int): _description_
            actor_obs_space (MultiSpace): _description_
            action_space (gym.spaces.Space): _description_
            store_device (_type_, optional): _description_. Defaults to "cuda:0".
            out_device (_type_, optional): _description_. Defaults to "cuda:0".
            n_envs (int, optional): _description_. Defaults to 1. 
            n_values (int, optional): _description_. Defaults to 1.
        """
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        assert self.action_size, "Action size must not be zero"
        

        self.n_values = n_values 
        self.horizon_length = horizon_length
        self.obs_space = obs_space
        
        self.store_device = store_device
        self.out_device = out_device
        self.n_envs = n_envs
        
        self.reset()
        pass
    
    def reset(self) -> None:
        """
        Create the buffers and set all initial values to zero
 
        """
        self.dones = torch.zeros(( self.n_envs, self.horizon_length,), dtype=torch.int8, device=self.store_device)
        self.values = torch.zeros((self.n_envs, self.horizon_length, self.n_values), dtype=torch.float32, device=self.store_device)
         
        # the mean of the action distributions
        self.action_mu = torch.zeros((self.n_envs, self.horizon_length, self.action_size), dtype=torch.float32, device=self.store_device)
        # the std(sigma) of the action distributions   
        self.action_std = torch.zeros((self.n_envs, self.horizon_length, self.action_size), dtype= torch.float32, device=self.store_device)
         
        
        self.obs = {}
        for key, obs_shape in self.obs_space.dict_shape.items():
            self.obs[key] = torch.zeros((self.n_envs, self.horizon_length) + obs_shape, dtype=torch.float32, device= self.store_device)
        
        # when the padding masks are true, the values will be disgarded
        self.src_key_padding_mask = torch.ones((self.n_envs, self.horizon_length), device= self.store_device, dtype=torch.bool)
        self.tgt_key_padding_mask = torch.ones((self.n_envs, self.horizon_length), device= self.store_device, dtype=torch.bool)
        
        
    def add(
        self, 
        obs: Dict[str, torch.Tensor],
        action_mu: torch.Tensor,
        action_std: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor):
        """Add one set of collected experience to the sequence buffer

        Args:
            obs (Dict[str, torch.Tensor]): Observations of the last step shape(num_envs, ) + obs_shape
            action_mu (torch.Tensor): mean of the current actions (num_envs, num_actions) 
            action_std (torch.Tensor): standard deviation of the current actions (num_envs, num_actions) 
            values (torch.Tensor): predicted value of the state (num_envs, num_values)
            dones (torch.Tensor): determines, whether the step was terminal for the episode (num_envs)
        """
        
        
        # roll the last to the first position
        
        # The tensord fill upd from 
        for key in self.obs:   
            self.obs[key] =  torch.roll(self.obs[key], shifts=(-1), dims=(1)) 
        
        self.action_mu = torch.roll(self.action_mu, shifts=(-1), dims=(1))
        self.action_std = torch.roll(self.action_std, shifts=(-1), dims=(1))
        self.values = torch.roll(self.values, shifts=(-1), dims=(1))
        self.dones = torch.roll(self.dones, shifts=(-1), dims=(1))
        
        self.src_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        self.tgt_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        
        for key in self.obs:   
            self.obs[key][:, -1, :] = obs[key].detach().to(self.store_device)
            
        self.action_mu[:, -1, :] = action_mu.detach().to(self.store_device)
        self.action_std[:, -1, :] = action_std.detach().to(self.store_device)
        self.values[:, -1,:] = values.detach().to(self.store_device)
        self.dones[:, -1] = dones.detach().to(self.store_device)
        
        
        # for every true dones at index 1 -> erase all old states to the left
        # and set the src and tgt key padding masks correctly
        last_dones = self.dones[:, -2].to(torch.bool)
        self.src_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        self.tgt_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        
        # only full row paddings have to be made, because the old padding still exist  
        additive_padding_dones_mask =  torch.unsqueeze((last_dones), 1).tile(1, self.horizon_length)
        additive_padding_dones_mask[:, -1] = torch.zeros_like(last_dones).to(torch.bool)
        
        self.src_key_padding_mask = torch.logical_or(self.src_key_padding_mask, additive_padding_dones_mask)
        self.tgt_key_padding_mask = torch.logical_or(self.tgt_key_padding_mask, additive_padding_dones_mask)

        
    def get_and_merge_last_obs(self, obs_dict: Dict[str, torch.Tensor], sequence_length: int) -> Dict[str, torch.Tensor]:
        """Gets the sequence length -1 amount of previous observations and merges them with the obs dict, given as input
        
        This function does not change the 
        Args:
            obs_dict (Dict[str, torch.Tensor]): observations of the last world iteration shape(num_envs, ) + obs _shape
            sequence_length (int): dim of output tensors => (n_envs,sequence_length, ) + obs_shape

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        
        for key in obs_dict.keys():
            res = self.obs[key][:, -(sequence_length-1)::, :] 
            obs_dict[key] = torch.concat((res, torch.unsqueeze(obs_dict[key], 1)), dim= 1)
        
        return obs_dict
    
    
    def get_last_sequence_step_data(self, sequence_length: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """returns the relevant data for the next step 

        Args:
            sequence_length (int): amount of time the agent can look into the past

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]: Dict containing relevant info 
        """
        obs_dict = {}
        for key in self.obs.keys():
            obs_dict[key] = self.obs[key][:, -(sequence_length)::, :] 
            
        sequence_action_mu = self.action_mu[:, -(sequence_length)::,:]
        sequence_action_std = self.action_std[:, -(sequence_length)::,:]
        sequence_value = self.values[:, -(sequence_length)::,:]
        seq_src_key_padding_mask = self.src_key_padding_mask[:, -(sequence_length)::, :]
        seq_tgt_key_padding_mask = self.tgt_key_padding_mask[:, -(sequence_length)::, :]
            
        return {
            'src_obs': obs_dict,
            'tgt_action_mu': sequence_action_mu,
            'tgt_action_std': sequence_action_std,
            'tgt_value': sequence_value,
            'src_pad_mask': seq_src_key_padding_mask,
            'src_tgt_pad_mask': seq_tgt_key_padding_mask
        }
    
        
class TransformerPPO:
    
    def __init__(self,
                 env: VecTask,
                 config: Dict,
                 device: Union[str, torch.device],
                 logger: BaseLogger,
                 policy: TransformerPolicy,
                 verbose: bool = True,
                 model_out_name: Optional[str] = None,
                 reward_to_beat_for_out: Optional[int] = None,
                 start_num_timesteps: int = 0,
                 start_epoch_num: int = 0
                 ) -> None:
         
        self.name = config['name']
        self.verbose = verbose
        self.logger = logger
        
        if isinstance(logger, TensorboardLogger) and hasattr(env, 'set_tensorboard_logger'):
            env.set_tensorboard_logger(logger)
            
        self.config = config
        self.env = env
        self.device = device
        self.env.set_simulation_log_callback(self.log_sim_parameters)  
        
        self.policy: TransformerPolicy = policy
        self.policy.to(self.device)
        
        self.num_envs = env.num_envs  
        
        self.seq_len = self.config.get('seq_length', 4)
         
        self.value_size = config.get('value_size',1)
        self.obs_space: MultiSpace = env.observation_space          
        self.action_space: gym.spaces.Space = env.action_space
        
        self.weight_decay = config.get('weight_decay', 0.0)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')
        self.learning_rate = config['learning_rate']
         
        
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.linear_lr:
            self.scheduler = LinearScheduler(float(config['learning_rate']), 
                max_steps=self.max_epochs, 
                apply_to_entropy=config.get('schedule_entropy', False),
                start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler = IdentityScheduler()
            
        self.max_epochs = self.config.get('max_epochs', 1e6)
             
        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        
        self.horizon_length = config['horizon_length']
        
        self.last_1000_ep_reward = deque([], maxlen=1000)
        
        self.max_epochs = self.config.get('max_epochs', 1e6)
         
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.linear_lr:
            self.scheduler = LinearScheduler(float(config['learning_rate']), 
                max_steps=self.max_epochs, 
                apply_to_entropy=config.get('schedule_entropy', False),
                start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler = IdentityScheduler()
            
            
        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        
        # sequence length refers to the length of the sequence ingested by the transformer network
        self.sequence_length = self.config['sequence_length']
        
        # horizon length is the amount of data after which the agent begins training 
        self.horizon_length = config['horizon_length']
          
          
        reward_shaper_config= config.get('reward_shaper', {})
        
        self.reward_shaper = DefaultRewardsShaper(
            scale_value=reward_shaper_config.get('scale_value', 1),
            shift_value=reward_shaper_config.get('shift_value', 0),
            min_val= reward_shaper_config.get('min_val', -np.Inf),
            max_val= reward_shaper_config.get('max_val', np.Inf)
            )
         
        self.normalize_advantage = config['normalize_advantage'] 
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)
        
        
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)
            
        
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.gae_lambda = self.config['gae_lambda']
        
        self.batch_size = self.horizon_length * self.num_envs
        
        self.minibatch_size = self.config['minibatch_size']
        
        if self.minibatch_size == 'max':
            self.minibatch_size = self.batch_size
        
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        
        assert(self.batch_size % self.minibatch_size == 0), "The Batch size must be divisible by the minibatch_size"
        
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        self.last_lr = self.config['learning_rate']
        
        self.entropy_coef = self.config['entropy_coef']
        
        self.value_bootstrap = self.config.get('value_bootstrap')
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        self.epoch_num = start_epoch_num
        self.play_time = 0
        self.update_time = 0
        
        # the total amount of timesteps playes across all environments
        self.num_timesteps  = start_num_timesteps
        
        # the average reward that was the highest as an average over an epoch play
        self.most_avg_reward_received = 0 
        
        if reward_to_beat_for_out is not None:
            self.reward_to_beat_for_out = float(reward_to_beat_for_out)
        else:
            self.reward_to_beat_for_out = None
        
        self.model_out_name = model_out_name
        
        
        
        # from the continuois A2cBaseAlgorithm
        self.is_discrete = False
        self.bounds_loss_coef = config.get('bounds_loss_coef', 0.001)
        
        self.actions_num = self.action_space.shape[0]
        self.clip_actions = config.get('clip_actions', True)
        
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        
        self.best_episode_reward = -np.inf
        self.last_lr = float(self.last_lr)
        
        # TODO add appropriate Dataset
        
         
    def init_tensors(self):
        
        self.sequence_buffer = SequenceBuffer(
            sequence_length=self.sequence_length,
            horizon_length= self.horizon_length,
            obs_space=self.obs_space,
            action_space=self.action_space,
            store_device=self.device,
            out_device=self.device,
            n_envs= self.num_envs,
            n_values=self.value_size
        )
         
        reward_shape = (self.num_envs, self.value_size)
        self.current_rewards = torch.zeros(reward_shape, dtype=torch.float32, device= self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)
        self.current_dones = torch.ones((self.num_envs, ), dtype=torch.uint8, device=self.device)
        
         
    def set_eval(self):
        self.policy.eval() 
        if self.normalize_value:
            self.value_mean_std.eval() 

    def set_train(self):
        self.policy.train() 
        if self.normalize_value:
            self.value_mean_std.train()
            
             
    def discount_values(self, 
                        dones: torch.Tensor,
                        extrinsic_values: torch.Tensor, 
                        rewards: torch.Tensor,
                        prev_dones: torch.Tensor,
                        prev_extrinsic_value: torch.Tensor) -> torch.Tensor:
        """Calculate the discounted values 

        Args:
            dones (torch.Tensor): Shape(horizon_length, n_envs)
            extrinsic_values (torch.Tensor): shape(horizon_length, n_envs, num_values)
            rewards (torch.Tensor): shape(horizon_length, n_envs)
            prev_dones (torch.Tensor): (n_envs)
            prev_extrinsic_value (torch.Tensor): (n_envs, num_values)

        Returns:
            torch.Tensor: shape 
        """

        lastgaelam = 0
        mb_advs = torch.zeros_like(rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - prev_dones
                nextvalues = prev_extrinsic_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - prev_extrinsic_value[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        return mb_advs
              
        
    def log_sim_parameters(self, sim_logs: Dict[str, Any])-> None:
        """Log parameters given by the simulation with the logger

        Args:
            sim_logs (Dict[str, Any]): simulation logs
        """
        
        for key, val in sim_logs.items():
            self.logger.log('sim/'+ key, val, step=self.num_timesteps)
        
        
    def play_steps(self):
        
        step_time = 0.0
        
        # cumulative sum of  episode rewards within rollout ()
        sum_ep_reward = 0
        
        # cumulative sum of  episode objective rewards within rollout ()
        sum_ep_objective_reward = 0
        
        # cumulative sum of the amount of completed episodes
        n_completed_episodes = 0
        
        # cumulative sum of all the steps taken in all the episodes
        sum_steps_per_episode = 0
        
        # the cumulative reweard constituents, if they exist
        sum_reward_consituents = {}
        
        step_reward = 0
        
        for n in range(self.horizon_length):
            
            res_dict = self.get_action_values(self.obs)
            
     
    def get_action_values(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            obs (Dict[str, torch.Tensor]): observation dict 

        Returns:
            Dict[str, torch.Tensor]: result dict with keys: 'value', 'mu'
        """
        self.policy.eval()
        
        with torch.no_grad():
            res = self.policy(is_train= False, obs= obs, prev_actions= None)
        
        if self.normalize_value:
            res['values'] = self.value_mean_std(res['values'], True)
        
        return res