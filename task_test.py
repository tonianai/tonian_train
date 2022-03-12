


from warnings import resetwarnings
from tonian.tasks.walking.walking_task import WalkingTask

import gym 
import numpy as np

from tonian.algorithms.ppo import PPO
from tonian.policies.policies import SimpleActorCriticPolicy

from gym.spaces import space
from tonian.tasks.base.command import Command
from  tonian.tasks.base.vec_task import MultiSpace, VecTask
from tonian.common.schedule import Schedule

import gym
from gym import spaces
import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole

import torch

import yaml



 
env = WalkingTask(config_or_path={"env": {"num_envs": 1000}}, sim_device="gpu" , graphics_device_id=0 , headless=False)
 
#env = Cartpole(config_or_path={"env": {"num_envs": 10}}, sim_device="gpu", graphics_device_id=0, headless=False)

env.is_symmetric = False
 
config_path = "./tonian/algorithms/ppo_config.yaml"

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
                                 lr_schedule=lr_schedule,
                                 init_log_std = 0.0,
                                 actor_hidden_layer_sizes=( 64, 64),
                                 critic_hiddent_layer_sizes=(64, 64),
                                 device="cuda:0")
 


algo = PPO(env, config, policy=policy, device="cuda:0")

# train for a million steps
algo.learn(total_timesteps=1e10)

# show the learned policy

print("I have learned")

env.close()

