from gym.spaces import space
from tonian.tasks.common.command import Command
from tonian.tasks.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

import torch

from tonian.tasks.cartpole.cartpole_task import Cartpole

 
env = Cartpole(config_path="./tonian/tasks/cartpole/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)

env.reset()


print(env.actor_observation_spaces.shape)


for i in range(10000): 
    
    
    action = torch.tensor([ env.action_space.sample() for _ in range(env.num_envs)])
    # command 
    # state 
    
    #print(action)
    
    
    #action = np.ones((num_agents, env.get_action_size()))
    obss, rewards, dones, _ =   env.step(action)
    
    print("actor")
    print(obss[0]["linear"].shape)
    print("critic")
    print(obss[1]["linear"].shape)
    
    
    if i % 1000 == 0:
        env.reset()
     

