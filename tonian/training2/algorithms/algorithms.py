from typing import Dict, Tuple, Any, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

from tonian.tasks.base.vec_task import VecTask
from tonian.common.logger import BaseLogger
from tonian.common.spaces import MultiSpace
from tonian.training2.common.schedulers import AdaptiveScheduler, LinearScheduler, IdentityScheduler
from tonian.training2.policies import A2CBasePolicy
from tonian.training2.common.helpers import DefaultRewardsShaper
from tonian.training2.common.running_mean_std import RunningMeanStd, RunningMeanStdObs
from tonian.training2.common.buffers import DictExperienceBuffer
from tonian.training2.common.common_losses import critic_loss, actor_loss
from tonian.training2.common.dataset import PPODataset

from tonian.common.utils import join_configs

import torch.nn as nn
import torch, gym, os, yaml, time


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


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

class A2CBaseAlgorithm(ABC):
    
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
        
        self.logger = logger
        
        self.config = config
        self.env = env
        self.device = device
        
        self.frame = 0
        
        self.policy = policy
        self.policy.to(self.device)
        
        self.num_envs = env.num_envs
        self.num_actors = env.get_num_actors_per_env()
        
        
        self.seq_len = self.config.get('seq_length', 4)
        
        
        self.value_size = config.get('value_size',1)
        self.actor_obs_space: MultiSpace = env.actor_observation_spaces
        self.critic_obs_space: MultiSpace = env.critic_observation_spaces
        
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
            max_val= reward_shaper_config.get('max_val', np.Inf)
            )
        
        
        self.normalize_input = self.config['normalize_input']
        self.normalize_advantage = config['normalize_advantage'] 
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)
        
        
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.device)
            
        if self.normalize_input:
            
            self.actor_obs_mean_std = RunningMeanStdObs(self.actor_obs_space.dict_shape).to(self.device)
            if self.critic_obs_space:
                self.critic_obs_mean_std = RunningMeanStdObs(self.critic_obs_space.dict_shape).to(self.device)
        
        
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
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        self.epoch_num = 0
        self.play_time = 0
        self.update_time = 0
        

        
        
    def set_eval(self):
        self.policy.eval() 
        if self.normalize_value:
            self.value_mean_std.eval()
        if self.normalize_input:
            self.actor_obs_mean_std.eval()
            self.critic_obs_mean_std.eval()

    def set_train(self):
        self.policy.train() 
        if self.normalize_value:
            self.value_mean_std.train()
        
        if self.normalize_input:
            self.actor_obs_mean_std.train()
            self.critic_obs_mean_std.train()    
        
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
            n_actors= self.num_actors,
            n_values=self.value_size )
    
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
    def calc_gradients(self):
        pass
    
    @abstractmethod
    def update_epoch(self):
        pass
    
    def get_action_values(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Optional[Dict[str, torch.Tensor]]):
        self.policy.eval()
        
        with torch.no_grad():
            proc_actor_obs, proc_critic_obs = self._preproc_obs(actor_obs, critic_obs)
            
            res = self.policy(is_train= False, actor_obs= proc_actor_obs, critic_obs= proc_critic_obs, prev_actions= None)
        
        if self.normalize_value:
            res['values'] = self.value_mean_std(res['values'], True)
        
        return res
        
    def _preproc_obs(self, actor_obs_batch: Dict[str, torch.Tensor], critic_obs_batch: Optional[Dict[str, torch.Tensor]]):
        
        res_actor_obs_batch = None
        res_critic_obs_batch = None
        
        if self.normalize_input:
            
            res_actor_obs_batch = self.actor_obs_mean_std(actor_obs_batch)
            if critic_obs_batch:
                res_critic_obs_batch = self.critic_obs_mean_std(critic_obs_batch)
            
            return  res_actor_obs_batch, res_critic_obs_batch
        
        return actor_obs_batch, critic_obs_batch
        
        
        
        
    def get_values(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Dict[str, torch.Tensor]):
        
        # TODO Do this faster 
        with torch.no_grad():
            self.policy.eval()
            result = self.policy(is_train= False, actor_obs= actor_obs, critic_obs= critic_obs, prev_actions= None)
            
            value = result['values']
            
            if self.normalize_value:
                value = self.value_mean_std(value, True)
            
            return value
            
    def train_epoch(self):
        pass
    
    
    def update_lr(self, lr):
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
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
            
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, reward_constituents = self.env.step(actions)
 
        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return obs, rewards.to(self.device), dones.to(self.device), infos, reward_constituents

    
    def play_steps(self):
        
        
        step_time = 0.0
        
        for n in range(self.horizon_length):
            
            res_dict = self.get_action_values(self.actor_obs, self.critic_obs)
            
            
            self.experience_buffer.update_value('critic_obs', n, self.critic_obs)
            self.experience_buffer.update_value('actor_obs', n, self.actor_obs)
            self.experience_buffer.update_value('dones', n, self.dones.detach().clone())
            
            neglogpacs = res_dict['neglogpacs']
            values = res_dict['values']
            actions = res_dict['actions']
            mus = res_dict['mus']
            sigmas = res_dict['sigmas']
            self.experience_buffer.update_value('neglogpacs', n, neglogpacs.detach().clone())
            self.experience_buffer.update_value('values', n, values.detach().clone())
            self.experience_buffer.update_value('actions', n, actions.detach().clone())
            self.experience_buffer.update_value('mus', n, mus.detach().clone())
            self.experience_buffer.update_value('sigmas', n, sigmas.detach().clone())
            
            
            step_time_start = time.time()
            
            obs, rewards, self.dones, infos, reward_constituents = self.env_step(actions)
            
            if isinstance(obs, Tuple):
                self.actor_obs = obs[0]
                if len(obs) == 2:
                    self.critic_obs = obs[1]
            else:
                self.actor_obs = obs 
            
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            
            shaped_rewards = self.reward_shaper(rewards)
            
            if self.value_bootstrap and 'time_outs' in infos:
                
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
                
            
            self.experience_buffer.update_value('rewards', n, shaped_rewards)
            
            
            self.current_rewards += rewards
            self.current_lengths += 1
            
            # todo add rewards and legnth stats and log all the writer stats
            
            
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
        last_values = self.get_values(self.actor_obs, self.critic_obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        
        tensor_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'states', 'dones', 'critic_obs', 'actor_obs']
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
            
        return batch_dict
    
class ContinuousA2CBaseAlgorithm(A2CBaseAlgorithm, ABC):
    
    def __init__(self, env: VecTask, config: Dict, device: Union[str, torch.device], logger: BaseLogger, policy: A2CBasePolicy) -> None:
        super().__init__(env, config, device, logger, policy)

        self.is_discrete = False
        self.bounds_loss_coef = config.get('bounds_loss_coef', 0.001)
        
        self.actions_num = self.action_space.shape[0]
        self.clip_actions = config.get('clip_actions', True)
        
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        
        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, False, self.device, self.seq_len)
        
        
    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        return rescaled_actions
    
    def train_epoch(self):
        super().train_epoch()
        
        self.set_eval()
        
        play_time_start = time.time()
        
        with torch.no_grad():
            batch_dict = self.play_steps()
        
        play_time_end = time.time()
        update_time_start = play_time_end
        
        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        
        a_losses = [] # actor losses
        c_losses = [] # critic losses
        b_losses = [] # boudning losses
        entropies = [] # entropy losses
        kls = [] # kl divergences 
        
        for _ in range(0,self.mini_epochs_num):
            
            ep_kls = []
            
            for i in range(len(self.dataset)):
                
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)
                    
                    
                self.dataset.update_mu_sigma(cmu, csigma)   
                
                if self.schedule_type == 'legacy':  
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl.item())
                    self.update_lr(self.last_lr)
                
            av_kls = torch.mean(torch.stack(ep_kls))  
                
            if self.schedule_type == 'standard': 
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)
                
        if self.schedule_type == 'standard_epoch':
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,av_kls.item())
            self.update_lr(self.last_lr)
            
        
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        
        # print(f" Play TIme is {play_time}")
        update_time = update_time_end - update_time_start
        # print(f" Update TIme is {update_time} N_Epochs {self.mini_epochs_num}")
        total_time = update_time_end - play_time_start
        

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
            
    def train(self):
        
        self.init_tensors()
        
        self.actor_obs, self.critic_obs = self.env_reset()
        
        total_time = 0
        
        while True:
            epoch_num = self.update_epoch()
            
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses,  entropies, kls, last_lr, lr_mul = self.train_epoch()
            
            print(kls)
            
            total_time += sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            total_time += sum_time
            
            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
             
            
            frame = self.frame
            
            scaled_time = sum_time #self.num_agents * sum_time
            scaled_play_time = play_time #self.num_agents * play_time
            
            self.logger.log('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
            self.logger.log('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
            self.logger.log('performance/step_fps', curr_frames / step_time, frame)
            self.logger.log('performance/rl_update_time', update_time, frame)
            self.logger.log('performance/step_inference_time', play_time, frame)
            self.logger.log('performance/step_time', step_time, frame)
            self.logger.log('losses/a_loss', torch.mean(torch.stack(a_losses)).item(), frame)
            self.logger.log('losses/c_loss', torch.mean(torch.stack(c_losses)).item(), frame)
            self.logger.log('losses/entropy', torch.mean(torch.stack(entropies)).item(), frame)
            self.logger.log('info/last_lr', last_lr * lr_mul, frame)
            self.logger.log('info/lr_mul', lr_mul, frame)
            self.logger.log('info/e_clip', self.e_clip * lr_mul, frame)
            self.logger.log('info/kl', torch.mean(torch.stack(kls)).item(), frame)
            self.logger.log('info/epochs', epoch_num, frame)
            # TODO Add Logging and shit
            
            
        
            
    def prepare_dataset(self, batch_dict: Dict[str, torch.Tensor]):
        """Prepare the local dataset for an epoch
        -> also calculate the advantages and normalize values and advantages if desired
        Args:
            batch_dict (Dict[str, torch.Tensor]): batch dict collected during the play_steps function
        """
        
        #  tensor_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'states', 'dones', 'critic_obs', 'actor_obs']
        # returns also, since they are added later in the play_steps function
        
        critic_obs = batch_dict['critic_obs']
        actor_obs = batch_dict['actor_obs']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        
        advantages = returns - values
        
        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
        
        advantages = torch.sum(advantages, axis=1)
        
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['critic_obs'] = critic_obs
        dataset_dict['actor_obs'] = actor_obs
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        
        self.dataset.update_values_dict(dataset_dict)
        
    @abstractmethod
    def train_actor_critic(self):
        pass
        
         
class PPOAlgorithm(ContinuousA2CBaseAlgorithm):
    
    def __init__(self, env: VecTask, config: Dict, device: Union[str, torch.device], logger: BaseLogger, policy: A2CBasePolicy) -> None:
        super().__init__(env, config, device, logger, policy)
        
        self.last_lr = float(self.last_lr)
        
        self.has_value_loss = True
        
        # TODO Implement input normalization
        
        
    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
    
    def calc_gradients(self, 
                       value_preds_batch: torch.Tensor, 
                       old_action_log_probs_batch: torch.Tensor,
                       advantage: torch.Tensor,
                       old_mu_batch: torch.Tensor,
                       old_sigma_batch: torch.Tensor,
                       return_batch: torch.Tensor,
                       actions_batch: torch.Tensor,
                       actor_obs_batch: Dict[str, torch.Tensor],
                       critic_obs_batch: Dict[str, torch.Tensor]
                       ):
        
        lr_mul = 1.0
        
        curr_e_clip = lr_mul * self.e_clip
        
        actor_obs_batch, critic_obs_batch = self._preproc_obs(actor_obs_batch, critic_obs_batch)
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.policy(is_train= True, actor_obs= actor_obs_batch, critic_obs = critic_obs_batch, prev_actions = actions_batch)
             
            
            action_log_probs = res_dict['prev_neglogprob']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            
            
            a_loss = actor_loss(old_action_log_probs_batch, action_log_probs, advantage, True,  curr_e_clip)
            
            if self.has_value_loss:
                c_loss = critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.device)
                
            a_loss = a_loss.unsqueeze(1)
            b_loss = self.bound_loss(mu).unsqueeze(1)
            entropy_loss = entropy.unsqueeze(1)
            
            
            a_loss = torch.mean(a_loss)
            b_loss = torch.mean(b_loss)
            c_loss = torch.mean(c_loss)
            entropy = torch.mean(entropy_loss)
            
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            for param in self.policy.parameters():
                param.grad = None
            
        self.scaler.scale(loss).backward()
        
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm)
            
        self.scaler.step(self.optimizer)
        self.scaler.update()    
        
        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, True)
        
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def train_actor_critic(self, input_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]):
        self.calc_gradients(value_preds_batch = input_dict['old_values'], 
                            old_action_log_probs_batch = input_dict['old_logp_actions'],
                            advantage = input_dict['advantages'],
                            old_mu_batch = input_dict['mu'],
                            old_sigma_batch = input_dict['sigma'],
                            return_batch = input_dict['returns'],
                            actions_batch = input_dict['actions'],
                            actor_obs_batch = input_dict['actor_obs'],
                            critic_obs_batch = input_dict['critic_obs'])
        return self.train_result
        
    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    