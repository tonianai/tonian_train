


from warnings import resetwarnings
from elysium.tasks.walking.walking_task import WalkingTask

import gym 
import numpy as np

from elysium.algorithms.ppo import PPO
from elysium.algorithms.policies import SimpleActorCriticPolicy

from gym.spaces import space
from elysium.tasks.base.command import Command
from  elysium.tasks.base.vec_task import MultiSpace, VecTask 

import numpy as np
from elysium.tasks.cartpole.cartpole_task import Cartpole, CartpoleSb3Task

import torch

import yaml




env = CartpoleSb3Task(config_path="./elysium/tasks/cartpole/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False, rl_device = "cpu")


import gym

from stable_baselines3 import PPO
import numpy as np



print(env.action_space)
print(env.observation_space)


model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)


obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()