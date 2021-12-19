from gym.spaces import space
from  envs.base.vec_task import MultiSpace, VecTask
import gym
from gym import spaces
import numpy as np

from envs.walking.walking_task import WalkingTask

num_obs = 10

visual_size = (100, 100)

spaces = MultiSpace(
    {'linear': spaces.Box(low=-1.0, high=1.0, shape=(num_obs, )),
     'visual': spaces.Box(low=-1.0, high=1.0, shape=visual_size)})



walking_task = WalkingTask(config_path="./envs/walking/config.yaml", sim_device="gpu", graphics_device_id=0, headless=False)