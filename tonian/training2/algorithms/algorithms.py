from typing import Dict, Tuple, Any, Union
from abc import ABC, abstractmethod
import numpy as np

from tonian.tasks.base.vec_task import VecTask
from tonian.common.logger import BaseLogger
from tonian.common.spaces import MultiSpace
from tonian.training2.common.schedulers import AdaptiveScheduler, LinearScheduler, IdentityScheduler
from tonian.training2.policies import A2CBasePolicy
from tonian.training2.common.helpers import DefaultRewardsShaper
from tonian.training2.common.running_mean_std import RunningMeanStd


import torch, gym

class BaseAlgorithm(ABC):
    
    def __init__(self,
                 env: VecTask,
                 config: Dict,
                 device: Union[str, torch.device],
                 logger: BaseLogger,
                 policy: A2CBasePolicy
                 ) -> None:
        
        self.name = config['name']
        
        self.config = config
        self.env = env
        self.device = device
        
        self.policy = policy
        
        self.num_actors = env.num_envs * env.get_num_actors_per_env()
        
        self.actor_obs_space: MultiSpace = env.actor_obs
        self.critic_obs_space: MultiSpace = env.critic_obs
        
        self.weight_decay = config.get('weight_decay', 0.0)
        
        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')
         
        
        self.learning_rate = config['learning_rate']
        
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
        
        self.horizon_length = config['horizon_length']
        
        reward_shaper_config= config.get('reward_shaper', {})
        self.reward_shaper = DefaultRewardsShaper(
            scale_value=reward_shaper_config.get('scale_value', 1),
            shift_value=reward_shaper_config.get('shift_value', 0),
            min_val= reward_shaper_config.get('min_val', -np.Inf),
            max_val= reward_shaper_config.get('max_val', -np.Inf)
            )
        
        
        self.normalize_advantage = config['normalize_advantage'] 
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)
        
        
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        
        
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.gae_lambda = self.config['gae_lambda']
        
        self.batch_size = self.horizon_length * self.num_actors 
        
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        
        
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        self.last_lr = self.config['learning_rate']
        
        
        self.entropy_coef = self.config['entropy_coef']
        
        
        self.value_bootstrap = self.config.get('value_bootstrap')
        
        
    def set_eval(self):
        self.policy.eval() 
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.policy.train() 
        if self.normalize_value:
            self.value_mean_std.train()