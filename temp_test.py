from gym.spaces import space
from  envs.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

from envs.walking.walking_env import WalkingEnv



env = WalkingEnv(config_path="./envs/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)


env.reset()

print("============ output space ==================")
print(env.action_space)

print("=========== critic input spaces ==============")
print(env.critic_input_spaces)


print("=========== actor input spaces ================")
print(env.actor_input_spaces)

for i in range(10000): 
    
    
    action = env.action_space.sample()
    
    print(action)
    
    
    #action = np.ones((num_agents, env.get_action_size()))
    env.step(action)
    
    if i % 1000 == 0:
        env.reset()
     

