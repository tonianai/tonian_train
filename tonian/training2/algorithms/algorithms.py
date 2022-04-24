from typing import Dict, Tuple, Any, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

from tonian.tasks.base.vec_task import VecTask
from tonian.common.logger import BaseLogger
from tonian.common.spaces import MultiSpace
from tonian.training2.common.schedulers import AdaptiveScheduler, LinearScheduler, IdentityScheduler
from tonian.training2.policies import A2CBasePolicy
from tonian.training2.common.helpers import DefaultRewardsShaper
from tonian.training2.common.running_mean_std import RunningMeanStd
from tonian.training2.common.buffers import DictExperienceBuffer

from tonian.common.utils import join_configs


import torch, gym, os, yaml

class BaseAlgorithm(ABC):
    
    def __init__(self,
                 env: VecTask,
                 config: Dict,
                 device: Union[str, torch.device],
                 logger: BaseLogger,
                 policy: A2CBasePolicy
                 ) -> None:
        
        base_config = self.get_standard_config()
        
        config = join_configs(config, base_config)
        
        self.name = config['name']
        
        self.config = config
        self.env = env
        self.device = device
        
        self.policy = policy
        
        self.num_envs = env.num_envs
        self.num_actors = env.get_num_actors_per_env()
        
        
        self.value_size = config.get('value_size',1)
        self.actor_obs_space: MultiSpace = env.actor_obs
        self.critic_obs_space: MultiSpace = env.critic_obs
        
        self.action_space: gym.spaces.Space = env.action_space
        
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
        
        self.batch_size = self.horizon_length * self.num_actors * self.num_envs
        self.batch_size_envs = self.horizon_length * self.num_actors
        
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        
        
        assert(self.batch_size % self.minibatch_size == 0), "The Batch size must be divisible by the minibatch_size"
        
        
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        self.last_lr = self.config['learning_rate']
        
        
        self.entropy_coef = self.config['entropy_coef']
        
        
        self.value_bootstrap = self.config.get('value_bootstrap')
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        
        
    def set_eval(self):
        self.policy.eval() 
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.policy.train() 
        if self.normalize_value:
            self.value_mean_std.train()
            
    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        return mb_advs
            
    def init_tensors(self):
        
        batch_size = self.num_envs * self.num_actors
        self.experience_buffer = DictExperienceBuffer(
            self.horizon_length, 
            self.critic_obs_space, 
            self.actor_obs_space, 
            self.action_space,
            store_device=self.device,
            out_device=self.device,
            n_envs=self.num_envs,
            n_actors= self.num_actors)
    
        reward_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(reward_shape, dtype=torch.float32, device= self.device)
        self.current_lengths = torch.zeros(batch_size, dtype= torch.float32, device= self.device)
        self.dones = torch.ones((batch_size, ), dtype=torch.uint8, device=self.device)

    def env_reset(self):
        return self.env.reset()
    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()
    
    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError()
    
    @abstractmethod
    def learn(self, n_steps: int, 
              verbose: bool = True, 
              early_stopping: bool = False,
              early_stopping_patience: int = 1e8, 
              reset_num_timesteps: bool = True):
        raise NotImplementedError()
    
    @abstractmethod
    def calc_gradients(self):
        pass
    
    @abstractmethod
    def update_epoch(self):
        pass
    
    def get_action_values(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Optional[Dict[str, torch.Tensor]]):
        self.policy.eval()
        
        with torch.no_grad():
            res = self.policy(actor_obs, critic_obs)
        
    
    def get_standard_config(self) -> Dict:
        """Retreives the standard config for the algo

        Returns:
            Dict: config
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_base_algo.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            
    def play_steps(self):
        
        for n in range(self.horizon_length):
            pass
            
            
    