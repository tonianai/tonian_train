
from typing import Dict, Tuple, Any, Union, Optional, List
from abc import ABC, abstractmethod
from collections import deque
import numpy as np

from tonian_train.tasks import VecTask
from tonian_train.common.logger import BaseLogger, TensorboardLogger
from tonian_train.common.spaces import MultiSpace
from tonian_train.common.schedulers import AdaptiveScheduler, LinearScheduler, IdentityScheduler
from tonian_train.policies import SequentialPolicy
from tonian_train.common.helpers import DefaultRewardsShaper
from tonian_train.common.running_mean_std import RunningMeanStd, RunningMeanStdObs 
from tonian_train.common.common_losses import critic_loss, actor_loss, calc_dynamics_loss
from tonian_train.common.utils import join_configs
from tonian_train.common.sequence_buffer import SequenceBuffer, SequenceDataset
from tonian_train.common.torch_utils import tensor_dict_clone
from tonian_train.common.obs_saver import ObservationSaver

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
    
  

class SequentialPPO:
    
    def __init__(self,
                 env: VecTask,
                 config: Dict,
                 device: Union[str, torch.device],
                 logger: BaseLogger,
                 policy: SequentialPolicy,
                 verbose: bool = True,
                 model_out_name: Optional[str] = None,
                 reward_to_beat_for_out: Optional[int] = None,
                 start_num_timesteps: int = 0,
                 start_epoch_num: int = 0
                 ) -> None:
        """ PPO Algorithm with sequential input as well as masks to the network

        Args:
            env (VecTask): _description_
            config (Dict): _description_
            device (Union[str, torch.device]): _description_
            logger (BaseLogger): _description_
            policy (TransformerPolicy): _description_
            verbose (bool, optional): _description_. Defaults to True.
            model_out_name (Optional[str], optional): _description_. Defaults to None.
            reward_to_beat_for_out (Optional[int], optional): _description_. Defaults to None.
            start_num_timesteps (int, optional): _description_. Defaults to 0.
            start_epoch_num (int, optional): _description_. Defaults to 0.
        """
         
        self.name = config['name']
        self.verbose = verbose
        self.logger = logger
        self.save_obs = False
        self.obs_saver = None
        
        if isinstance(logger, TensorboardLogger) and hasattr(env, 'set_tensorboard_logger'):
            env.set_tensorboard_logger(logger)
            
        self.config = config
        self.env = env
        self.device = device
        self.env.set_simulation_log_callback(self.log_sim_parameters)  
        
        self.policy: SequentialPolicy = policy
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
        
        
        self.dynamics_coef = config.get('dynamics_coef', 1.0)
        self.has_dynamics_loss = config.get('has_dynamics_loss', False)
        
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
            self.value_mean_std = RunningMeanStd((1,), is_sequence=True).to(self.device)
            
        
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
        
        self.has_value_loss = True  
        
         
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
        
    
    def train(self, max_steps:Optional[int] = None, save_obs: bool = False):
        
        self.init_tensors()
        self.save_obs = save_obs
        if self.save_obs:
            self.obs_saver = ObservationSaver("saved_obs")
            self.obs_saver.clear_base_path()
        
        self.obs = self.env.reset()
        
        total_time = 0
        
        epoch_num = 0
        
        while True:
            
            time_dict, a_losses, c_losses, b_losses, d_losses, entropies, kls, last_lr, lr_mul, actor_sigma = self.train_epoch()
            
            total_time += time_dict['total_time']
            epoch_num += 1
            steps_per_second = self.batch_size / (time_dict['play_time'] + time_dict['model_update_time'])
                               
            self.logger.log("z_speed/steps_per_second", steps_per_second , self.num_timesteps)
            self.logger.log("z_speed/time_on_rollout", time_dict['play_time'], self.num_timesteps)
            self.logger.log("z_speed/time_on_train", time_dict['model_update_time'], self.num_timesteps)
            self.logger.log("z_speed/step_time", time_dict['step_time'], self.num_timesteps)
            self.logger.log("z_speed/time_on_dataset_creation", time_dict['dataset_creation_time'], self.num_timesteps)
            self.logger.log("z_speed/time_fraq_on_rollout", time_dict['play_time'] / ( time_dict['total_time']), self.num_timesteps)
            self.logger.log("z_speed/time_fraq_on_train", time_dict['model_update_time'] / (time_dict['total_time']), self.num_timesteps)
            self.logger.log("z_speed/time_frac_on_dataset", time_dict['dataset_creation_time'] / (time_dict['total_time']), self.num_timesteps)

            self.logger.log('losses/a_loss', torch.mean(torch.stack(a_losses)).item(), self.num_timesteps)
            self.logger.log('losses/c_loss', torch.mean(torch.stack(c_losses)).item(), self.num_timesteps)
            self.logger.log('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.num_timesteps)
            self.logger.log('info/last_lr', last_lr * lr_mul, self.num_timesteps)
            self.logger.log('info/lr_mul', lr_mul, self.num_timesteps)
            self.logger.log('info/actor_sigma',  torch.mean(torch.stack(actor_sigma)).item(), self.num_timesteps)
            self.logger.log('info/e_clip', self.e_clip * lr_mul, self.num_timesteps)
            self.logger.log('info/kl', torch.mean(torch.stack(kls)).item(), self.num_timesteps)
            self.logger.log('info/epochs', epoch_num, self.num_timesteps)
            if len(b_losses) > 0:
                self.logger.log('losses/bounds_loss', torch.mean(torch.stack(b_losses)), self.num_timesteps)
                
            if self.has_dynamics_loss:
                self.logger.log('losses/dynamics_loss', torch.mean(torch.stack(d_losses)), self.num_timesteps)
            
            self.logger.update_saved()
            
            if self.verbose: 
                print(" Run: {}    |     Iteration: {}     |    Steps Trained: {:.3e}     |     Steps per Second: {:.0f}     |     Time Spend on Rollout: {:.2%}".format(self.logger.identifier ,epoch_num, self.num_timesteps, steps_per_second, time_dict['play_time'] / ( time_dict['total_time'])))
            
            if max_steps is not None and max_steps < self.num_timesteps:
                break;
                
                
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
        
        
        self.set_train()
        
        dataset = SequenceDataset( buffer=self.sequence_buffer ,minibatch_size= self.minibatch_size, 
                                  runnign_mean_value= self.value_mean_std)
    
        data_creation_time_end = time.time()
        
        a_losses = [] # actor losses
        c_losses = [] # critic losses
        b_losses = [] # boudning losses
        d_losses = [] # dynamics losses
        entropies = [] # entropy losses
        
        actor_sigmas = []
        
        kls = [] # kl divergences 
        
        for _ in range(0,self.mini_epochs_num):
            ep_kls = []
            
            for i in range(len(dataset)):
                data = dataset[i]
                
                a_loss, c_loss,  entropy, d_loss, kl, last_lr, lr_mul, cmu, actor_sigma, b_loss = self.calc_gradients(data)    
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                d_losses.append(d_loss) 
                ep_kls.append(kl)
                entropies.append(entropy)
                actor_sigmas.append(actor_sigma)
                
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss) 
                    
                # todo: add mu and sigma to dataset maybe
                    
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
         
        
        time_dict = {
            'step_time': res_dict['step_time'],
            'play_time': play_time_end - play_time_start,
            'dataset_creation_time': data_creation_time_end - play_time_end,
            'model_update_time': update_time_end  - data_creation_time_end,
            'total_time': update_time_end - play_time_start
            
            
        }
        
        return  time_dict, a_losses, c_losses, b_losses, d_losses,  entropies, kls, last_lr, lr_mul, actor_sigmas
    
                                                     
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
        
        list_steps_per_episode = []
        
        # the cumulative reweard constituents, if they exist
        sum_reward_consituents = {}
        
        step_reward = 0
        
        for _ in range(self.horizon_length):
            self.policy.eval()
            
            last_obs = tensor_dict_clone(self.obs)
            last_dones = self.dones.detach().clone()
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
            
            if self.save_obs and self.obs_saver:
                with torch.no_grad():
                    normalized_obs = self.policy.normalize_obs(last_obs)
                self.obs_saver.maybe_save_obs(normalized_obs)
             
            step_time_start = time.time()
            
            self.obs, rewards, self.dones, infos, reward_constituents = self.env_step(actions)
            
            
            
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)
            
            shaped_rewards = self.reward_shaper(rewards)
            
            if self.value_bootstrap and 'time_outs' in infos:
                
                shaped_rewards += self.gamma * values * infos['time_outs'].unsqueeze(1).float()
    
            self.sequence_buffer.add(obs=last_obs,
                                     next_obs=self.obs,
                            action= actions,
                            rewards=shaped_rewards,
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
            
            list_steps_per_episode.extend( infos["episode_steps"][torch.nonzero(infos["episode_steps"])][:,0].tolist())
            
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
        
         
        self.sequence_buffer.calc_advantages(last_dones, last_value, self.gamma, self.gae_lambda)
         
        if "objective_episode_reward" in infos: 
            self.logger.log("run/last_1000_obj_ep_reward", sum(self.last_1000_ep_reward)/1000, self.num_timesteps)
        
       
        # this dict will be returned from the play steps function, to facilitate training
        result_dict = {} 
        result_dict['played_frames'] = self.batch_size
        result_dict['step_time'] = step_time
        self.sequence_buffer.block_write()
        
        
        step_reward = torch.sum(rewards) / (self.num_envs * self.horizon_length)
        
        self.logger.log("run/step_reward", step_reward, self.num_timesteps)
        
        
        # --- log the results before exiting
        
        if n_completed_episodes != 0:
            self.logger.log("run/episode_rewards", sum_ep_reward / n_completed_episodes, self.num_timesteps)
            
            self.logger.log("run/objective_episode_rewards", sum_ep_objective_reward / n_completed_episodes, self.num_timesteps)
            
            self.logger.log("run/steps_per_episode", sum_steps_per_episode / n_completed_episodes, self.num_timesteps)
        
            if sum_ep_objective_reward == 0:
                self.current_avg_reward = sum_ep_reward / n_completed_episodes
            else:
                self.current_avg_reward = sum_ep_objective_reward / n_completed_episodes
                
            steps_per_episode_tensor = torch.FloatTensor(list_steps_per_episode)
            self.logger.log("step_per_ep/std",steps_per_episode_tensor.std() , self.num_timesteps)
            self.logger.log("step_per_ep/mean",steps_per_episode_tensor.mean() , self.num_timesteps) 
            self.logger.log("step_per_ep/01_quantile", torch.quantile(steps_per_episode_tensor,0.01) , self.num_timesteps)
            self.logger.log("step_per_ep/10_quantile", torch.quantile(steps_per_episode_tensor,0.1) , self.num_timesteps)
            self.logger.log("step_per_ep/50_quantile", torch.quantile(steps_per_episode_tensor,0.5) , self.num_timesteps)
            self.logger.log("step_per_ep/90_quantile", torch.quantile(steps_per_episode_tensor,0.9) , self.num_timesteps)
            self.logger.log("step_per_ep/99_quantile", torch.quantile(steps_per_episode_tensor,0.99) , self.num_timesteps)
                
                
            if self.current_avg_reward > self.most_avg_reward_received:
                self.most_avg_reward_received = self.current_avg_reward
                self.save(best_model = True)
            self.save(best_model= False)
            
            if sum_reward_consituents:
                # log the reward constituents
                for key, value in sum_reward_consituents.items():
                    self.logger.log(f"run_reward_{key}", value / self.horizon_length, self.num_timesteps )
            
        
        
        
        return result_dict
        
                 
    def calc_gradients(self, input_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor]:
        """calculate the gradients and all losses 

        Args:
            input_dict (Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]): _description_

        Returns:
            Tuple[torch.Tensor]: _description_
        """
        
        
        lr_mul = 1.0
        
        curr_e_clip = lr_mul * self.e_clip
        
        with torch.cuda.amp.autocast( enabled=self.mixed_precision):
            
            # the mask is necessary, because the agent sees otherwise sees the action made by the actor given the most recent observation
            tgt_mask = self.policy.get_tgt_mask(self.sequence_length)
            res_dict = self.policy.forward(is_train= True, 
                                               src_obs= input_dict['obs'], 
                                               tgt_action_mu= input_dict['action_mu'], 
                                               tgt_action_std= input_dict['action_std'],
                                               tgt_value= input_dict['values'],
                                               prev_actions= input_dict['action'],
                                               src_padding_mask= input_dict['src_key_padding_mask'],
                                               tgt_padding_mask= input_dict['tgt_key_padding_mask'])
            
            
            action_log_probs = res_dict['prev_neglogprob']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            next_state_pred = res_dict['next_state_pred']
            
            old_action_log_probs_batch = input_dict['neglogprobs'][:, -1]
            advantage = torch.squeeze(input_dict['advantages'][:, -1])
            value_preds_batch = input_dict['values'][:, -1]
            return_batch = input_dict['returns'][: , -1]
            
            old_mu_batch = input_dict['action_mu'][: , -1]
            old_sigma_batch = input_dict['action_std'][: , -1]
             
            
            if self.has_dynamics_loss:   
                next_obs = {key: value[:, -1] for key, value in input_dict['next_obs'].items()} # only the most recent observation is used for the next state prediction
                predicted_obs = res_dict['next_state_pred']
                with torch.no_grad():
                    next_obs = self.policy.normalize_obs(next_obs, override_training=True, training_value=False )
                    
                dynamics_loss = calc_dynamics_loss( predicted_obs, next_obs)
                dynamics_loss = dynamics_loss.unsqueeze(1)
                dynamics_loss = torch.mean(dynamics_loss) 
            
            else:
                dynamics_loss = torch.zeros(1, device=self.device)

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
             
            c_loss = c_loss * self.critic_coef * 0.5
            entropy = entropy * self.entropy_coef
            b_loss = b_loss * self.bounds_loss_coef
            d_loss = dynamics_loss * self.dynamics_coef 
             
            
            loss = a_loss + c_loss - entropy + b_loss + d_loss 
            
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
         
        
        return (a_loss, c_loss, entropy, d_loss,\
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
        
        
    def save(self, best_model: bool = True):
        """Save the model to the file system

        Args:
            best_model (bool, optional): _description_. Defaults to True.
        """
        if best_model:
            run_save_dir = os.path.join(self.logger.folder, 'saves', 'best_model')
            print("save best policy")
        else:
            run_save_dir = os.path.join(self.logger.folder, 'saves', 'last_model')
            
            
        os.makedirs(run_save_dir, exist_ok=True)
        
        self.policy.save(run_save_dir)
        
        torch.save(self.optimizer.state_dict(),  os.path.join(run_save_dir, 'optim.pth'))
        
        if self.normalize_value:
            torch.save(self.value_mean_std.state_dict(), os.path.join(run_save_dir, 'value_mean_std.pth'))
            
        if self.save_obs and self.policy.obs_normalizer:
            torch.save(self.policy.obs_normalizer.state_dict(), os.path.join('obs_normalizer', 'obs_normalizer.pth'))
    
        
        if self.model_out_name :
            if best_model and self.reward_to_beat_for_out and self.most_avg_reward_received < self.reward_to_beat_for_out:
                return
            
            save_dir = os.path.join('models', self.model_out_name)
 
            os.makedirs(save_dir, exist_ok=True)
            
            # register the model unter the given name 
            self.policy.save(save_dir)
           
            if self.normalize_value:
                torch.save(self.value_mean_std.state_dict(), os.path.join(save_dir, 'value_mean_std.pth'))  
              
    def load(self, path: str):
        """Load the model from the file system

        Args:
            path (str): _description_
        """
         
        self.policy.load(path)
        
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optim.pth')))
          
        if self.normalize_value:
            self.value_mean_std.load_state_dict(torch.load(os.path.join(path, 'value_mean_std.pth')))
            
    
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
        """Take a step in the environment

        Args:
            actions (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, reward_constituents = self.env.step(actions)
 
        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return obs, rewards.to(self.device), dones.to(self.device), infos, reward_constituents
    
    
    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    
    def update_lr(self, lr):
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr