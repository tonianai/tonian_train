import numpy
import os
import torch
import torch.nn as nn

import numpy as np


from gym import spaces
import gym

import yaml
import time
import os

from typing import Dict, Optional, Any, Union


from tonian.common.spaces import MultiSpace


from isaacgym.torch_utils import to_torch
from isaacgym import gymutil, gymtorch, gymapi


from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from tonian.tasks.base.vec_task import VecTask, BaseEnv, GenerationalVecTask

class Cartpole(VecTask):

    def __init__(self, config_or_path: Optional[Union[str, Dict]], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0"):
        super().__init__(config_or_path, sim_device, graphics_device_id, headless, rl_device)
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    

        self.reset_dist = self.config["env"]["resetDist"]

        self.max_push_effort = self.config["env"]["maxEffort"]
        self.max_episode_length = 500

        self.config["env"]["numObservations"] = 4
        self.config["env"]["numActions"] = 1

        
        # aquire the state tensor of the dof
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
    
    
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
        
        assert self.config["sim"] is not None, "The sim config must be set on the task config file"
        assert self.config["env"] is not None, "The env config must be set on the task config file"
        
        #extract params from config 
        assert self.config["env"]["powerscale"]
        self.power_scale = self.config["env"]["powerscale"]
           
 

    def _create_envs(self, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)


        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"


        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)
        
        print(cartpole_asset)

        pose = gymapi.Transform()
        pose.p.z = 2.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.actor_obs["linear"][:, 2]
        pole_vel = self.actor_obs["linear"][:, 3]
        cart_vel = self.actor_obs["linear"][:, 1]
        cart_pos = self.actor_obs["linear"][:, 0]
         

        self.rewards, self.do_reset = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.do_reset, self.num_steps_in_ep, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
         

        self.actor_obs["linear"][env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.actor_obs["linear"][env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.actor_obs["linear"][env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.actor_obs["linear"][env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.actor_obs

    def reset_envs(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.do_reset[env_ids] = 0
        self.num_steps_in_ep[env_ids] = 0

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.num_steps_in_ep += 1

        env_ids = self.do_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_envs(env_ids)

        self.compute_observations()
        self.compute_reward()
        
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_obs = 5
        return MultiSpace({
            "linear": spaces.Box(low=-1.0, high=1.0, shape=(num_obs, ))
        })
        
    def _get_critic_observation_spaces(self) -> MultiSpace:
        """
        There is no critic observation space, this is a symemtric env
        """
        return self._get_actor_observation_spaces()
    
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        
        # just a single action, (this is cartpole)
        num_actions = 1
        return spaces.Box(low=-1.0, high=1.0, shape=(num_actions, )) 
    
    def _is_symmetric(self) -> bool:
        """This is a symmetric environment, that means, that the actor and critic observation spaces are shared
        Returns:
            [bool]
        """
        return True
    
    def reward_range(self):
        return (-1e100, 1e100)
    
    def close(self):
        pass

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # eslint-disable-next-line
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    
    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    
    

    return reward, reset

