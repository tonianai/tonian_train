from gym.spaces import space
from elysium.tasks.base.command import Command
from elysium.tasks.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

import torch

from elysium.tasks.walking.walking_task import WalkingTask

 
env = WalkingTask(config_path="./elysium/tasks/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)

env.reset()


print(env.actor_observation_spaces.shape)


for i in range(10000): 
    
    
    action = torch.tensor([ env.action_space.sample() for _ in range(env.num_envs)])
    # command 
    # state 
    
    #print(action)
    
    
    #action = np.ones((num_agents, env.get_action_size()))
    obss, rewards, dones, _ =   env.step(action)
    
    
    if i % 1000 == 0:
        env.reset()
     

