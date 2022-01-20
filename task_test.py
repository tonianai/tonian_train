


from warnings import resetwarnings
from elysium.tasks.walking.walking_task import WalkingTask

import gym 
import numpy as np

from elysium.algorithms.ppo import PPO
from elysium.algorithms.policies import SimpleActorCriticPolicy

from gym.spaces import space
from elysium.tasks.base.command import Command
from  elysium.tasks.base.vec_task import MultiSpace, VecTask
from elysium.common.schedule import Schedule

import gym
from gym import spaces
import numpy as np
from elysium.tasks.cartpole.cartpole_task import Cartpole

import torch

import yaml



 
#env = WalkingTask(config_path="./elysium/tasks/walking/config.yaml", sim_device="gpu" , graphics_device_id=0 , headless=False)
 
env = Cartpole(config_path="./elysium/tasks/cartpole/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)

env.is_symmetric = False

config_path = "./elysium/algorithms/ppo_config.yaml"

# opfen the config file 
with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:    
        raise FileNotFoundError( f"File {config_path} not found")
    
lr_schedule = Schedule(config["lr"])


policy = SimpleActorCriticPolicy(actor_obs_space=env.actor_observation_spaces,
                                 critic_obs_space=env.critic_observation_spaces,
                                 action_space= env.action_space,
                                 lr_schedule=lr_schedule)


algo = PPO(env, config, policy=policy, device="cuda:0")

# train for a million steps
algo.learn(total_timesteps=100000000)

# show the learned policy

print("I have learned")

env.close()

