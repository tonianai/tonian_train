from gym.spaces import space
from tasks.base.command import Command
from  tasks.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

import torch

from tasks.walking.walking_task import WalkingTask

 
env = WalkingTask(config_path="./tasks/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)

env.reset()


for i in range(10000): 
    
    
    action = torch.tensor([ env.action_space.sample() for _ in range(env.num_envs)])
    # command 
    # state 
    
    #print(action)
    
    
    #action = np.ones((num_agents, env.get_action_size()))
    env.step(action)
    
    if i % 1000 == 0:
        env.reset()
     

