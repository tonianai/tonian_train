
from abc import ABC, abstractmethod
import torch
import numpy as np

import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import datetime as datetime

from common.utils.type_aliases import Schedule

from tasks.base.vec_task import VecTask
from common.policies import BaseActorCritic, BasePolicy
from common.utils import get_device
from common.buffers import RolloutBuffer, RolloutBufferSamples



class BaseAlgorithm(ABC):
    
    def __init__(self,
                 policy: BasePolicy,
                 env: VecTask,
                 learning_rate: float,
                 device: str,
                 verbose: int
                 ) -> None:
        super().__init__()
        self.device = device
        self.env = env
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        self.policy = policy.to(self.device)
        
        
        
        self.action_size = env.action_size
        self.linear_obs_size = env.linear_obs_size
        self.visual_obs_shape = env.visual_obs_shape
        self.command_size = env.command_size
        
        self.num_agents = env.get_num_agents()
         
        
    def update_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        
      
    @abstractmethod
    def train(total_timesteps: int):
        raise NotADirectoryError
        
    @abstractmethod
    def load(checkpoint: str):
        raise NotImplementedError
    
    @abstractmethod
    def save(checkpont: str):
        raise NotImplementedError
        
        

class OnPolicyAlgorithm(BaseAlgorithm, ABC):
    
    def __init__(self, 
                 policy: BaseActorCritic,
                 env: VecTask, 
                 learning_rate: float, 
                 n_steps: int, 
                 gamma: float,  
                 gae_lambda: float,  
                 device: Union[torch.device, str],
                 verbose: int,
                 ) -> None:
        super().__init__(policy, env, learning_rate, device, verbose)
        self.n_steps = n_steps
        self.gamma = gamma   
        
        self.memory = RolloutBuffer(self.n_steps, env.linear_obs_size, env.visual_obs_shape, env.command_size, env.action_size, self.device, gae_lambda, gamma, env.num_agents)
        
        
        pass
    
    def train(self, total_steps: int, command: np.ndarray, update_freq: int = 200):
        
        
        if self.verbose > 1:
               start_time = datetime.now().replace(microsecond=0)
               print("Started training at (GMT) : ", start_time)
               print("=============================================")
        
        i_step = 0
        
        obs =  self.env.reset()
        linear_obs = obs[0]
        visual_obs = np.moveaxis(obs[1], -1, 1) # move the channels in front, as this is the shape that the conv nets require
        
        
        
        current_ep_reward = np.zeros(self.env.num_agents, dtype=float)
        
        
        while i_step < total_steps:
            
            # select an action 
            with torch.no_grad():
                actions, logprobs = self.select_action(linear_obs, visual_obs, command)
            
            new_obs, rewards, dones = self.env.step(actions)
            new_linear_obs = new_obs[0]
            new_visual_obs = np.moveaxis(new_obs[1], -1, 1) # move the channels in front, as this is the shape that the conv nets require
            
            
            
            
            #self.memory.add(linear_obs, visual_obs, command, actions, reward, episode_start, value, log_prob)
            
            
            linear_obs = new_linear_obs
            visual_obs = new_visual_obs
            
            i_step += 1
    
    
    def _learn(self):
        self.policy.train(True)
        
        self.policy.train()
        pass
    
    def select_action(self, linear_obs: np.ndarray, visual_obs: np.ndarray , command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select an action with the given input parameters

        Args:
            linear_obs (np.ndarray): expected shape (num_agents, linear_obs_size)
            visual_obs (np.ndarray): expected shape (num_agetns, ) + visual_obs_shape
            command (np.ndarray): expected shape (num_agents, command_size)
            
            
        Returns: 
        actions, logprobs
        
        """
        
        assert command.shape[0] == visual_obs.shape[0] == linear_obs.shape[0], "The batch size varies, between linear_obs, visual_obs and commands"
        
        with torch.no_grad():
            visual_obs = torch.from_numpy(visual_obs).float().to(self.device)
            linear_obs = torch.from_numpy(linear_obs).float().to(self.device)
            command = torch.from_numpy(command).float().to(self.device)
            
        
        
        return self.policy.act(linear_obs, visual_obs, command)
    
    
class PPO_Algorithm(OnPolicyAlgorithm):
    
    def __init__(self, 
                 policy: BaseActorCritic, 
                 env: VecTask, 
                 learning_rate: float  = 3e-4, 
                 n_steps: int = 1000, 
                 K_Epochs: int = 10,
                 eps_clip: float = 0.1,
                 action_std: Union[float, Schedule] = 0.5,
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95,
                 device: Union[torch.device, str] = "cpu", 
                 verbose: int = 1) -> None:
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, device, verbose)
        
        
    def load(checkpoint: str):
        return super().load()


    def save(checkpont: str):
        return super().save()
    
        
    
    
        