
from typing import Dict, Tuple, Any, Union, Optional, List
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

from torch.utils.data import Dataset

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
    
  


class SequenceBuffer():
    
    def __init__(self,
                 n_envs: int,  
                 horizon_length: int,
                 sequence_length: int, 
                 obs_space: MultiSpace,
                 action_space: gym.spaces.Space,
                 store_device: Union[str, torch.device] = "cuda:0",
                 out_device: Union[str, torch.device] = "cuda:0",
                 n_values: int = 1) -> None:
        """Buffer, that stores the observations and the outputs for the transformer sequence length
        
        TODO: fix that horizon length must be bigger than sequence length
        
        Args:
        
            horizon_length (int): the amount of steps taken in one epoch 
            sequence_length (int): the amount of steps the actor can look back
            obs_space (MultiSpace): the observation space 
            action_space (gym.spaces.Space): the action space
            store_device (_type_, optional): the device on which the tensors will be stored. Defaults to "cuda:0".
            out_device (_type_, optional): the device on which the tensors will be outputed. Defaults to "cuda:0".
            n_envs (int, optional): The amount of environments. Defaults to 1. 
            n_values (int, optional): The amount of values. Defaults to 1.
        """
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        assert self.action_size, "Action size must not be zero"
        

        self.n_values = n_values 
        self.horizon_length = horizon_length # The amount of steps played per epoch
        self.sequence_length = sequence_length # The amount of observations and actions taken into account by the network
        
        self.buffer_length = self.horizon_length + self.sequence_length # This buffer length is required, because 
        
        
        self.obs_space = obs_space
        
        self.store_device = store_device
        self.out_device = out_device
        self.n_envs = n_envs

        # The Sequence buffer can go in a state of refusing writes during learning        
        self.can_write = True
        
        
        # the data in these tensors comes from the outside
        self.external_tensor_names = ['dones', 'values', 'rewards', 'action', 'action_mu', 'action_std', 'neglogprobs', 'obs']
        
        # the data from these tensors is derived within the buffer
        self.derived_tensor_names = ['src_key_padding_mask', 'tgt_key_padding_mask', 'returns', 'advantages']
        
        
        self.tensor_names = self.external_tensor_names + self.derived_tensor_names
        
        # ----- Create the buffers and set all initial values to zero
 
        self.dones = torch.zeros(( self.n_envs, self.buffer_length,), dtype=torch.int8, device=self.store_device)
        self.values = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        
        self.rewards = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
     
        self.action = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype=torch.float32, device=self.store_device)
        
        # the mean of the action distributions
        self.action_mu = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype=torch.float32, device=self.store_device)
        # the std(sigma) of the action distributions   
        self.action_std = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype= torch.float32, device=self.store_device)
         
        self.neglogprobs = torch.zeros((self.n_envs, self.buffer_length), dtype= torch.float32, device=self.store_device)
     
        self.advantages = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        self.returns = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        
         
        
        self.obs = {}
        for key, obs_shape in self.obs_space.dict_shape.items():
            self.obs[key] = torch.zeros((self.n_envs, self.buffer_length) + obs_shape, dtype=torch.float32, device= self.store_device)
        
        # when the padding masks are true, the values will be disgarded
        self.src_key_padding_mask = torch.ones((self.n_envs, self.buffer_length), device= self.store_device, dtype=torch.bool)
        self.tgt_key_padding_mask = torch.ones((self.n_envs, self.buffer_length), device= self.store_device, dtype=torch.bool)
          
        
        # the advantage pointer always refer to the second (1) dimesnion, that correspons to the buffer length
        self.left_advantage_pointer = self.buffer_length -1 # the pointer at which the first correct advantages are
        self.right_advantage_pointer = self.buffer_length -1 # the pointer at which the last correct advantages are
  
    def add(
        self, 
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        action_mu: torch.Tensor,
        action_std: torch.Tensor,
        rewards: torch.Tensor, 
        values: torch.Tensor,
        dones: torch.Tensor,
        neglogprobs: torch.Tensor):
        """Add one set of collected experience to the sequence buffer

        Args:
            obs (Dict[str, torch.Tensor]): Observations of the last step shape(num_envs, ) + obs_shape
            action_mu (torch.Tensor): mean of the current actions (num_envs, num_actions) 
            action_std (torch.Tensor): standard deviation of the current actions (num_envs, num_actions) 
            values (torch.Tensor): predicted value of the state (num_envs, num_values)
            dones (torch.Tensor): determines, whether the step was terminal for the episode (num_envs)
        """
        
        assert self.can_write, "The Sequence buffer cannot be written to while the can_write flag is set to false"
        
        # roll the last to the first position
        
        # The tensord fill upd from 
        for key in self.obs:   
            self.obs[key] =  torch.roll(self.obs[key], shifts=(-1), dims=(1)) 
        
        self.action = torch.roll(self.action, shifts=(-1), dims=(1))
        self.action_mu = torch.roll(self.action_mu, shifts=(-1), dims=(1))
        self.action_std = torch.roll(self.action_std, shifts=(-1), dims=(1))
        self.values = torch.roll(self.values, shifts=(-1), dims=(1))
        self.dones = torch.roll(self.dones, shifts=(-1), dims=(1))
        self.neglogprobs = torch.roll(self.neglogprobs, shifts=(-1), dims = (1))
        self.rewards = torch.roll(self.rewards, shifts=(-1), dims=(1))
        
        # advantage and returns will also be rolled and set to 0, so that they align with the rest of the data
        self.advantages = torch.roll(self.advantages, shifts=(-1), dims=(1))
        self.returns = torch.roll(self.returns, shifts=(-1), dims=(1))
        
        self.advantages[:, -1,:] = torch.zeros_like(values, device= self.store_device)
        self.returns[:, -1,:] = torch.zeros_like(values, device= self.store_device)
        
        # both pointers get deducted one, because no additional data was added, the data was just shifted down by one
        # pointers can also not be smaller than 0
        self.left_advantage_pointer -= 1
        if self.left_advantage_pointer <= 0:
            self.left_advantage_pointer = 0
        self.right_advantage_pointer -= 1 
        if self.right_advantage_pointer <= 0:
            self.right_advantage_pointer = 0
        
        
        self.src_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        self.tgt_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        
        for key in self.obs:   
            self.obs[key][:, -1, :] = obs[key].detach().to(self.store_device)
             
            
        self.action[:, -1, :] = action.detach().to(self.store_device)
        self.action_mu[:, -1, :] = action_mu.detach().to(self.store_device)
        self.action_std[:, -1, :] = action_std.detach().to(self.store_device)
        self.values[:, -1,:] = values.detach().to(self.store_device)
        self.dones[:, -1] = dones.detach().to(self.store_device)
        self.neglogprobs[:, -1] = neglogprobs.detach().to(self.store_device)
        self.rewards[:, -1, : ] = rewards.detach().to(self.store_device)
        
        # for every true dones at index 1 -> erase all old states to the left
        # and set the src and tgt key padding masks correctly
        last_dones = self.dones[:, -2].to(torch.bool)
        self.src_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        self.tgt_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        
        # only full row paddings have to be made, because the old padding still exist  
        additive_padding_dones_mask =  torch.unsqueeze((last_dones), 1).tile(1, self.buffer_length)
        additive_padding_dones_mask[:, -1] = torch.zeros_like(last_dones).to(torch.bool)
        
        self.src_key_padding_mask = torch.logical_or(self.src_key_padding_mask, additive_padding_dones_mask)
        self.tgt_key_padding_mask = torch.logical_or(self.tgt_key_padding_mask, additive_padding_dones_mask)

        
    def get_and_merge_last_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Gets the sequence length -1 amount of previous observations and merges them with the obs dict, given as input
        
        This function does not change the obs buffer and is meant as a direct output for the trnsformer, hence the output in the shape of the sequence length
        Args:
            obs_dict (Dict[str, torch.Tensor]): observations of the last world iteration shape(num_envs, ) + obs _shape
    
        Returns:
            Dict[str, torch.Tensor]: dim of output tensors => (n_envs,sequence_length, ) + obs_shape

        """
        
        for key in obs_dict.keys():
            res = self.obs[key][:, -(self.sequence_length-1)::, :] 
            obs_dict[key] = torch.concat((res, torch.unsqueeze(obs_dict[key], 1)), dim= 1)
        
        return obs_dict
    
    
    def get_last_sequence_step_data(self, obs: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """returns the relevant data for the next step 

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]: Dict containing relevant info 
        """
        obs_dict = {}
        for key in self.obs.keys():
            res = self.obs[key][:, -(self.sequence_length)::, :]  
            obs_dict[key] = torch.concat((res, torch.unsqueeze(obs[key], 1)), dim= 1)
            
        sequence_action_mu = self.action_mu[:, -(self.sequence_length)::,:]
        sequence_action_std = self.action_std[:, -(self.sequence_length)::,:]
        sequence_value = self.values[:, -(self.sequence_length)::,:]
        seq_src_key_padding_mask = self.src_key_padding_mask[:, -(self.sequence_length)::]
        seq_src_key_padding_mask = torch.concat((seq_src_key_padding_mask,torch.unsqueeze(torch.zeros((self.n_envs, ), device = self.out_device).to(torch.bool), 1)),1)
            
        seq_tgt_key_padding_mask = self.tgt_key_padding_mask[:, -(self.sequence_length)::]
            
        return {
            'src_obs': obs_dict,
            'tgt_action_mu': sequence_action_mu,
            'tgt_action_std': sequence_action_std,
            'tgt_value': sequence_value,
            'src_padding_mask': seq_src_key_padding_mask,
            'tgt_padding_mask': seq_tgt_key_padding_mask
        }
    
    
    def get_reversed_order(self, tensor_names: Optional[List[str]] = None  ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Return every tensor in the reversed order
        -> the smallest timestep will be at 0 
           and the largest timestep will be at horizon_length
           
         
        Returns:
            Dict[str, torch.Tensor]: keys() => {'action', ''}
        """
        if tensor_names is None:
            tensor_names = self.tensor_names
            
        res_dict = {}
        for tensor_name in tensor_names:
            
            curr_tensor = getattr(self, tensor_name) # could be the obs dict
            
            if isinstance(curr_tensor, Dict):
                res_dict[tensor_name] = {obs_name: curr_tensor[obs_name][:, -self.horizon_length::] for obs_name in curr_tensor.keys()}
            else:
                res_dict[tensor_name] = torch.flip( curr_tensor[:, -self.horizon_length::], (1,))
            
        if len(tensor_names) == 1:
            return res_dict[tensor_name]
        return res_dict
            
            
    def calc_advantages(self, 
                        final_dones: torch.Tensor, 
                        final_pred_values: torch.Tensor,
                        gamma: float,
                        gae_lambda: float
                        ):
        """Discout the values and calculate the advantages for the horizon_len and add to the buffer

        Args:
            final_dones (torch.Tensor): shape (num_envs)
            final_pred_values (torch.Tensor): shape(num_envs, num_values) 
            gamma (float): discount factor
            gae_lambda (float): bootstrapping tradeoff
        """
        assert (self.buffer_length -1) - self.right_advantage_pointer == self.horizon_length, "The complete horizon length must be played before the advantage will be calculated"
        
        last_gae_lam = 0
        
        for i in range(self.horizon_length + 1):
            t = self.right_advantage_pointer + i # index in the uncalculated territory
            if i == 0:
                next_non_terminal = 1.0 - final_dones
                next_values = final_pred_values
            else:
                next_non_terminal = 1.0 - self.dones[:, t-1]
                next_values = self.values[:, t-1]
            next_non_terminal = next_non_terminal.unsqueeze(1)
                         
            # discounted differenct between last predicted values and current predicted values + reward
            delta = self.rewards[:, t] + gamma * next_values * next_non_terminal - self.values[:, t]    
            self.advantages[:, t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        
        # calculate returns 
        self.returns[:, self.right_advantage_pointer::] = self.advantages[:, self.right_advantage_pointer::] + self.values[:, self.right_advantage_pointer::]
        
        self.right_advantage_pointer = self.buffer_length -1
        
         
         
    def block_write(self):
        self.can_write = False
        
    def allow_write(self):
        self.can_write = True
        
        
    def __len__(self):
        return self.buffer_length
        
class SequenceDataset(Dataset):
    
    def __init__(self, data: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
                 n_envs: int,
                 horizon_len: int ,
                 sequence_len: int,
                 minibatch_size: int) -> None:
         
        
        self.sequence_length = sequence_len
        self.horizon_length = horizon_len
        
        self.minibatch_size = minibatch_size    
        
        self.length = n_envs * horizon_len // self.minibatch_size 
          
        self.data_buffer = {}
        buffer_res_dict : Dict[str, Union[Dict, torch.Tensor]] = data
        
        for key in buffer_res_dict.keys():
            
            if isinstance(buffer_res_dict[key], Dict):
                # obs dict
                self.data_buffer[key] = {obs_key : SequenceDataset.expand_and_compacify_tensor(buffer_res_dict[key][obs_key]) for obs_key in buffer_res_dict[key].keys()}
            else:
                # just tensor
                self.data_buffer[key] = SequenceDataset.expand_and_compacify_tensor(buffer_res_dict[key])
        pass
                
                 
    def expand_and_compacify_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Expand the shape of a tensor of shape (num_envs, horizon_length ,c, ...)
        to the shape (a, b, b, c, ...)
        and than compactify it into shape (a * b, b, c, ...)

        Args:
            tensor (torch.Tensor): Tensor of shape (a,b,c,...)

        Returns:
            torch.Tensor: Tensor of shape (a*b, b, c, ....)
        """
        
        # create the new tensor shape (-1, -1, b, -1 ***)
        expanded_shape = (-1,-1,)+ (tensor.shape[1],) + ((-1,) * (len(tensor.shape) - 2))  
        expanded_tensor = tensor.unsqueeze(2).expand(expanded_shape)
        
        # contract the first two dimensions
        
        compacted_shape = (expanded_tensor.shape[0] * expanded_tensor.shape[1], ) + tensor.shape[1:]
        return expanded_tensor.contiguous().view(compacted_shape)
        
        
        
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """get the minibatch at the specified indes 

        Args:
            index (int): index ranging from 0 to len(self) -1

        Returns:
            Dict[str, torch.Tensor]: torch(minibatch_size, sequence_length, ) + singluar_tensor_shape
        """
        
        start = index * self.minibatch_size
        end = start + self.minibatch_size
        
        out_dict = {}
        
        for key, buffer_item in self.data_buffer.items():
            if isinstance(buffer_item, Dict):
                out_dict[key] = {obs_key: buffer_item[obs_key][start: end] for  obs_key in buffer_item.keys()}
            else:
                out_dict[key] = buffer_item[start: end]
                
        
        return out_dict
        
        
    def __len__(self):
        return self.length
 
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
        
        self.seq_len = policy.sequence_length
         
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
        self.sequence_length = self.policy.sequence_length
        
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
        
        self.first_step = True 
        
        
        
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
            horizon_length= self.horizon_length,
            sequence_length= self.sequence_length,
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
        
        self.dones = torch.ones((self.num_envs, ), dtype=torch.uint8, device=self.device)
         
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
        
    
    def train(self, max_steps:Optional[int] = None):
        
        self.init_tensors()
        self.obs = self.env.reset()
        
        total_time = 0
        
        epoch_num = 0
        
        while True:
            
            self.train_epoch()
            
            epoch_num += 1
                              
                
                
    def train_epoch(self):
        """Play and train the policy once

        Returns:
            _type_: _description_
        """
        
        self.set_eval()
        
        play_time_start = time.time()
        
        
        with torch.no_grad():
            res_dict = self.play_steps()
        
        play_time_end = time.time()
        update_time_start = play_time_end
        
        
        data_to_sequence = self.sequence_buffer.get_reversed_order()
        # todo augment the data with the advantage gathered from the res_dict of the play steps function
        
        dataset = SequenceDataset(data= data_to_sequence, 
                                  n_envs= self.num_envs,
                                  horizon_len= self.horizon_length,
                                  sequence_len= self.sequence_length,
                                  minibatch_size= self.minibatch_size)
    
        self.set_train()
        
        a_losses = [] # actor losses
        c_losses = [] # critic losses
        b_losses = [] # boudning losses
        entropies = [] # entropy losses
        kls = [] # kl divergences 
        
        for _ in range(0,self.mini_epochs_num):
            ep_kls = []
            
            for i in range(len(dataset)):
                data = dataset[i]
                pass
                
                                                     
    def play_steps(self):
        """Play the environment for horizon_length amount of steps
        """
        self.sequence_buffer.allow_write() # The sequence buffer should only be written to during the play steps function
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
            self.policy.eval()
            
            last_obs = self.obs
            last_dones = self.dones
            last_data_dict = self.sequence_buffer.get_last_sequence_step_data(obs=last_obs)
            
            with torch.no_grad():
                res = self.policy.forward(is_train= False, prev_actions= None, **last_data_dict)
            
            if self.normalize_value:
                res['values'] = self.value_mean_std(res['values'], True)
            
            actions = res['actions']
            values = res['values']
            action_mus = res['mus']
            action_sigmas = res['sigmas']
            neglogprobs = res['neglogprobs']
            
             
            step_time_start = time.time()
            
            self.obs, rewards, self.dones, infos, reward_constituents = self.env_step(actions)
            
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            
            shaped_rewards = self.reward_shaper(rewards)
            
            if self.value_bootstrap and 'time_outs' in infos:
                
                shaped_rewards += self.gamma * values * infos['time_outs'].unsqueeze(1).float()
    
            self.sequence_buffer.add(obs=last_obs,
                            action= actions,
                            rewards=rewards,
                            action_mu=action_mus,
                            action_std=action_sigmas,
                            values = values,
                            dones= last_dones,
                            neglogprobs= neglogprobs)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            
            
            # add all the episodes that were completed whitin the last time step to the counter
            n_completed_episodes +=  torch.sum(self.dones).item()
            
            # sum of all rewards of all completed episodes
            sum_ep_reward += torch.sum(infos["episode_reward"]).item()
            
            if "objective_episode_reward" in infos:
                sum_ep_objective_reward += torch.sum(infos["objective_episode_reward"]).item()
                
                all_completed_ep_rewards = infos["objective_episode_reward"][torch.nonzero(infos["objective_episode_reward"])]
                
                for i in range(len(all_completed_ep_rewards)):
                    self.last_1000_ep_reward.append(all_completed_ep_rewards[i].item()) 
            
            # sum all the steps of all completed episodes
            sum_steps_per_episode  += torch.sum(infos["episode_steps"]).item()
            
            if not sum_reward_consituents:
                sum_reward_consituents = reward_constituents
            else:
                for key, value in reward_constituents.items():
                    sum_reward_consituents[key] += value    
            
            
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        self.num_timesteps += self.batch_size
        
        # ---- get the valus of the last state 
        with torch.no_grad():
            self.policy.eval()
            last_state_dict = self.sequence_buffer.get_last_sequence_step_data( obs=self.obs)
            result = self.policy.forward(is_train= False, **last_state_dict)
            
            last_value = result['values']
            
            if self.normalize_value:
                last_value = self.value_mean_std(last_value, True)
            
        
        # ---- calculate the advantages 
        
        last_dones = self.dones.float()
        buffer_result = self.sequence_buffer.get_reversed_order(['dones', 'values', 'rewards'])
        dones = buffer_result['dones'].float()
        values = buffer_result['values']
        rewards = buffer_result['rewards']
         
        self.sequence_buffer.calc_advantages(last_dones, last_value, self.gamma, self.gae_lambda)
         
        if "objective_episode_reward" in infos: 
            self.logger.log("run/last_1000_obj_ep_reward", sum(self.last_1000_ep_reward)/1000, self.num_timesteps)
        
       
        # this dict will be returned from the play steps function, to facilitate training
        result_dict = {} 
        result_dict['played_frames'] = self.batch_size
        result_dict['step_time'] = step_time
        self.sequence_buffer.block_write()
        
        
        step_reward = torch.sum(rewards) / (self.num_envs * self.horizon_length)
        
        # --- log the results before exiting
        
        if n_completed_episodes != 0:
            self.logger.log("run/episode_rewards", sum_ep_reward / n_completed_episodes, self.num_timesteps)
            
            self.logger.log("run/objective_episode_rewards", sum_ep_objective_reward / n_completed_episodes, self.num_timesteps)
            
            self.logger.log("run/steps_per_episode", sum_steps_per_episode / n_completed_episodes, self.num_timesteps)
        
            if sum_ep_objective_reward == 0:
                self.current_avg_reward = sum_ep_reward / n_completed_episodes
            else:
                self.current_avg_reward = sum_ep_objective_reward / n_completed_episodes
                
                
            if self.current_avg_reward > self.most_avg_reward_received:
                self.most_avg_reward_received = self.current_avg_reward
                self.save(best_model = True)
            self.save(best_model= False)
            
            if sum_reward_consituents:
                # log the reward constituents
                for key, value in sum_reward_consituents.items():
                    self.logger.log(f"run_reward_{key}", value / self.horizon_length, self.num_timesteps )
            
        
        
        
        return result_dict
        
                 
    def calc_gradients(input_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]):
        
        pass
        
        
    def save(self, best_model: bool = True):
        pass
            
    
    def preprocess_actions(self, actions: torch.Tensor):
        """preprocess the actions

        Args:
            actions torch.Tensor: 

        Returns:
            _type_: _description_
        """
        
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        return rescaled_actions
            
            
    def env_step(self, actions: torch.Tensor):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, reward_constituents = self.env.step(actions)
 
        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return obs, rewards.to(self.device), dones.to(self.device), infos, reward_constituents