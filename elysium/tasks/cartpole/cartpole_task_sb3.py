


from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from typing import Any, Callable, List, Optional, Sequence, Type, Union
from elysium.tasks.base.vec_task import VecTask, BaseEnv, GenerationalVecTask
import gym
import numpy as np



from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

from elysium.tasks.cartpole.cartpole_task import Cartpole


class CartpoleSb3Task(Cartpole, VecEnv):
    
    def __init__(self, config_path, sim_device, graphics_device_id, headless, rl_device: str = "gpu:0"):
        super().__init__(config_path, sim_device, graphics_device_id, headless, rl_device=rl_device)
        
    def close(self) -> None:
        return super().close()
    
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return super().env_is_wrapped(wrapper_class, indices=indices)
    
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
    
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return super().get_attr(attr_name, indices=indices)
    
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return super().seed(seed=seed)
    
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return super().set_attr(attr_name, value, indices=indices)
    
    
    def step_async(self, actions: np.ndarray) -> None:
        print("step async")
    
    def step_wait(self) -> VecEnvStepReturn:
        print("step async")