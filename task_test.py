import gym 
import numpy as np

from stable_baselines3 import PPO

from gym.spaces import space
from elysium.tasks.base.command import Command
from  elysium.tasks.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

import torch

from elysium.tasks.walking.walking_task import WalkingTask

 
env = WalkingTask(config_path="./tasks/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)



model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)