import gym
from typing import Union, Iterable, Dict, Tuple, List, Optional, Sequence, Any, Type
from tonian.tasks.base.vec_task import VecTask
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import numpy as np

import torch
import torch.nn as nn


# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]

def _obs_to_numpy(obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        
    result = {}
    if isinstance(obs, Dict):
        for key, value in obs.items():
            result[key] = value.clone().cpu().detach().numpy()
            
    else:
         obs = obs.clone().cpu().detach().numpy()
        
    return result


class Sb3VecEnvWrapper(VecEnv):
    """Wraps the isaacgym vec env in an gym env -> this is highly inefficient and should only be used for testing
    """
    
    
    def __init__(self, vec_task: VecTask) -> None:
        
        assert vec_task.is_symmetric, "Non symmetric vec tasks cannot work with sb3 wrappers"
        
        super().__init__(vec_task.num_envs, vec_task.observation_space, vec_task.action_space)
        self.vec_task = vec_task
        pass
    
    
    
    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """         
        return   _obs_to_numpy(self.vec_task.reset())
       

    
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        self.actions = actions
    
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        # turn actions into an torch tensor
        torch_actions = torch.from_numpy(self.actions).to(self.vec_task.device)
         
        
        # make the step
        observations, rewards, dones, info = self.vec_task.step(torch_actions)
        
        # turns results from the step into numpy array
        observations = _obs_to_numpy(observations)
        rewards = rewards.clone().cpu().detach().numpy()
        dones = dones.clone().cpu().detach().numpy()
         
        return observations, rewards, dones, []

    
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        self.vec_task.close()
    
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        return [False] * self.vec_task.num_envs

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """ 

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """
        self.vec_task.render()

    
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        return [0] * self.vec_task.num_envs

    @property
    def unwrapped(self) -> "VecEnv":
        return self
        

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
    
    

            
    
    