

from tonian.tasks.mk1.mk1_walking.mk1_walking_task import Mk1WalkingTask
from tonian.tasks.walking.walking_task import WalkingTask

from warnings import resetwarnings
from tonian.common.logger import DummyLogger

from tonian.training.algorithms.ppo import PPO
from tonian.training.policies.policies import SimpleActorCriticPolicy


import gym
from gym import spaces
import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.common.utils import set_random_seed
import torch

import yaml, time



set_random_seed(40, True)

 
env = Mk1WalkingTask(config={"env": {"num_envs": 10, "reward_weighting": { "death_height": 0}}}, sim_device="gpu" , graphics_device_id=0 , headless=False)
 
#env = Cartpole(config_or_path={"env": {"num_envs": 10}}, sim_device="gpu", graphics_device_id=0, headless=False)

 
config_path = "./tonian/algorithms/ppo_config.yaml"

# opfen the config file 
with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:    
        raise FileNotFoundError( f"File {config_path} not found")
    
# lr_schedule = Schedule(config["lr"])
# 
# policy = SimpleActorCriticPolicy(actor_obs_space=env.actor_observation_spaces,
#                                  critic_obs_space=env.critic_observation_spaces,
#                                  action_space= env.action_space,
#                                  lr_schedule=lr_schedule,
#                                  init_log_std = 0.0,
#                                  actor_hidden_layer_sizes=( 64, 64),
#                                  critic_hiddent_layer_sizes=(64, 64),
#                                  device="cuda:0")
#  
# 
# 
# algo = PPO(env, config, policy=policy, device="cuda:0", logger= DummyLogger() )

# train for a million steps
# algo.learn(total_timesteps=1e10)

for _ in range(100):
    for i in range(100):
        ones_actions = torch.ones((env.num_envs, env.action_size)).to("cuda:0")  * 1
        env.step(ones_actions)
    time.sleep(30.0)
# show the learned policy

print("I have learned")

env.close()

