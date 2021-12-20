from gym.spaces import space
from  envs.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

from envs.walking.walking_env import WalkingEnv



env = WalkingEnv(config_path="./envs/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)

lol_space =  MultiSpace({
            "shitz": spaces.Box(low=-1.0, high=1.0, shape=(10, 10)),
            "giigles": spaces.Box(low=-1.0, high=1.0, shape=(100, ))
        })
 
 
lol_space.join_with(env.critic_observation_spaces)

print(lol_space)


env.reset()

print("============ output space ==================")
print(env.action_space)

print("=========== critic input spaces ==============")
print(env.critic_observation_spaces)


print("=========== actor input spaces ================")
print(env.actor_observation_spaces)

for i in range(10000): 
    
    
    action = env.action_space.sample()
    
    #print(action)
    
    
    #action = np.ones((num_agents, env.get_action_size()))
    env.step(action)
    
    if i % 1000 == 0:
        env.reset()
     

