




from warnings import resetwarnings
from tonian.tasks.walking.walking_task import WalkingTask

import gym 
import numpy as np

from stable_baselines3 import PPO 
 
from tonian.tasks.base.sb3_vecenv_wrapper import Sb3VecEnvWrapper

import gym
from gym import spaces
import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole

import torch

import yaml



 
#env = WalkingTask(config_or_path={'env': {'num_envs': 10}}, sim_device="gpu" , graphics_device_id=0 , headless=False)
 
env = Cartpole(config_or_path={'env': {'num_envs': 12}}, sim_device="gpu", graphics_device_id=0, headless=False)

env.is_symmetric = True


sb3_env = Sb3VecEnvWrapper(env)

model = PPO("MultiInputPolicy", sb3_env, verbose=1)

print(model.policy)
 
model.learn(total_timesteps=200000000)
