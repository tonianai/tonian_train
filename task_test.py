


from elysium.tasks.walking.walking_task import WalkingTask

import gym 
import numpy as np

from elysium.algorithms.ppo import PPO
from elysium.algorithms.policies import SimpleActorCriticPolicy

from gym.spaces import space
from elysium.tasks.base.command import Command
from  elysium.tasks.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np


import torch

import yaml




 
env = WalkingTask(config_path="./elysium/tasks/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)


config_path = "./elysium/algorithms/ppo_config.yaml"

# opfen the config file 
with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:    
        raise FileNotFoundError( f"File {config_path} not found")


policy = SimpleActorCriticPolicy(actor_obs_shapes=env.actor_observation_spaces.shape,
                                 critic_obs_shapes= env.critic_observation_spaces.shape, 
                                 action_size=env.action_space.shape[0],
                                 action_std_init=0.1, 
                                 actor_hidden_layer_sizes= (128, 128, 64),
                                 critic_hidden_layer_sizes= (128, 128, 61)
                                )


algo = PPO(env, config, policy=policy, device="cuda:0")

algo.learn(total_timesteps=10000)
