
from isaacgym import gymtorch, gymapi

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, Union, List, Optional, Tuple
from tonian.tasks2.agents.base_agents import BaseAgents
from tonian.common.utils.utils import dict_to_cpu
from tonian.common.utils.config_utils import join_configs
from tonian.common.spaces import MultiSpace

import numpy as np

import torch, os, yaml, gym, sys


class BaseTask(ABC):
    """
    Vectorized Environments are a method for stacking multiple independent environments into a single environment.
    Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step


    """
    
    def __init__(self, 
                 config: Optional[Dict[str]], 
                 device: Union[str, torch.device], 
                 graphics_device_id: int,
                 headless: bool, 
                 rl_device: Union[str, torch.device]):
        
        """An asymmetric actor, critic vectorized base simulation environment class based on isaacgym
        
        Args:     
            config (Dict[str, Any]): the config dictionary
            device (str): Device on which the environment runs ex: cuda:0, cuda or cpu
            headless (bool): determines whether the scene is rendered
            sim_device (str): Device of the output:  cuda:0, cuda or cpu
        """
        if config is None:
            config = {}
        
        # override the base config with values given in the config param
        self.config = join_configs(self._get_base_config(),  config)
        
        self.headless = headless
        self.device = device
        self.rl_device = rl_device
        
        self.device_id = 0
        
        # get the grapthics device id fromt the given device 
        device_split = self.device.split(":")
        if len(device_split) == 2:
            self.device_id = int(device_split[1])
        
        enable_camera_sensors = self.config.get("enableCameraSensors", False)        
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:   
            self.graphics_device_id = -1
            
        self.num_envs = self.config["num_envs"]
        
        # This implementation used Asymmetic Actor Critics
        # https://arxiv.org/abs/1710.06542
        self.critic_observation_spaces = self._get_critic_observation_spaces()
        self.actor_observation_spaces = self._get_actor_observation_spaces()
        self.action_space = self._get_action_space()
        self.metadata = {}
        
        
        self.physics_engine = gymapi.SIM_PHYSX
        
        self.sim_params = self._parse_sim_params(self.physics_engine, self.config["sim"])

        self.gym = gymapi.acquire_gym()
        
        self.is_sim_initialized = False
        self.sim = self._create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # create the environments 
        print(f'num envs {self.num_envs} env spacing {self.config["env"]["env_spacing"]}')
        self._create_envs( self.config["env"]["env_spacing"], int(np.sqrt(self.num_envs)))
        
        self.max_episode_length = self.config['env'].get('max_episode_length', 10000)
        
        # The Frequency with which the actions are polled relative to physics step
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)
        
        print(f"The max ep length is {self.max_episode_length}")
        
        self.is_sim_initialized = True
        
        self._allocate_buffers()
        
        self._set_viewer()
        
        # The amount of steps taken in this env as a whole
        self.global_step = 0
        
        
    
    def _create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()
            
        self._create_ground_plane(sim)
        

        return sim
    
    def _create_ground_plane(self, sim = None):
        if not sim:
            sim = self.sim
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        #plane_params.static_friction = self.plane_static_friction
        #plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(sim, plane_params)

    def _parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = True
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])
     

        # return the configured params
        return sim_params
            
    def _get_base_config(self) -> Dict:
        """Get the base configuration, values of which can be overwritten using the config parameter

        Returns:
            Dict: The base config of the Task, ase given by the base_config.yaml file
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'base_config.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
    
    def _allocate_buffers(self):
        """initialize the tensors on the gpu
          it is important, that these tensors reside on the gpu, as to reduce the amount of data, that has to be transmitted between gpu and cpu
        """
        
        
        # The reset buffer contains boolean values, that determine whether an environment of a given id should be reset in the next step
        # shape: (num_envs)
        self.do_reset = torch.zeros( (self.num_envs, ), device= self.device, dtype=torch.int8)
        
        
        # The timeout buffer is similar to the reset buffer
        # If the num_steps_in_ep exceeds the self.max_steps the value of the is_timeout is 1 at the position of that environment, and the env resets
        self.is_timeout = torch.zeros( (self.num_envs, ), device= self.device, dtype= torch.int8)
        
        """Note: The logical or between is_timout and do_reset determines whether the episode will be reset"""
        
        # The num_steps_in_ep buffer declares the amount of steps a enviromnent as done since 
        self.num_steps_in_ep = torch.zeros((self.num_envs, ) , device= self.device, dtype= torch.int32)
        
        # the masked numer of steos in episode, only shows the final value when the episode ternubates
        self.masked_num_steps_in_ep = torch.zeros((self.num_envs,), device= self.device, dtype= torch.int32)
        
        # This is a list of tensors, that reside on device, a list is needed, because we are dealing with multispace observations
        self.actor_obs = { key: torch.zeros((self.num_envs, ) + space_shape, device= self.device, dtype= torch.float32) for  (key, space_shape) in self.actor_observation_spaces.dict_shape.items()}

        
        if not self.is_symmetric:
            # only asymmetric environments have critic observations
            self.critic_obs = {key: torch.zeros((self.num_envs, ) + space_shape, device=self.device, dtype=torch.float32) for (key,space_shape) in self.critic_observation_spaces.dict_shape.items()}
        else:
            self.critic_obs = self.actor_obs
                   

        # Randomize buffer determines whether a given env should randomize 
        self.do_randomize = torch.zeros((self.num_envs, ), device=self.device, dtype= torch.int8 )
        
        # rewards tensor stores the rewards that were given at last
        self.rewards = torch.zeros((self.num_envs, ) , device= self.device, dtype=torch.float32)


        # the cumulative rewards achieved within the environment until this steop
        self.cumulative_rewards = torch.zeros(self.num_envs, device= self.device, dtype= torch.float32)
        
        # the cumulative rewards, but only the envs in terminal states are not hidden
        self.masked_cumulative_rewards = torch.zeros(self.num_envs, device= self.device, dtype= torch.float32)
        
    def _set_viewer(self):
        """Create the Viewer        """
        
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)
    
    def _render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)
    
    def step(self, actions: torch.Tensor) -> Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info
        """
        
        self.extras = {}
        
        self.pre_physics_step(actions)
        
        for i in range(self.control_freq_inv):
            # extra simulation steps
            self.render()
            self.gym.simulate(self.sim)
        
        #region timeout and resetting
        
        # increase the num_steps_in_ep tensor and reset environments if needed
        self.num_steps_in_ep += 1  
        # timeout buffer determines whether an episode exceedet its max_length of timesteps 
        self.is_timeout = torch.where(self.num_steps_in_ep >= self.max_episode_length - 1, torch.ones_like(self.is_timeout), torch.zeros_like(self.is_timeout))
        self.do_reset = self.is_timeout | self.do_reset

        # check if there are enviromments, that need a reset
        reset_env_ids = self.do_reset.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.reset_envs(reset_env_ids)
            
        # set the num steps in ep, the is_timeout and the do_reset tensors to 0
        self.num_steps_in_ep[reset_env_ids] = 0
         
        
        self.do_reset[reset_env_ids] = 0
        
        #endregion
        
        # Calculate obs and reward in the post pyhsics step (in the concrete implementation)
        self.post_physics_step()
        
        # add the rewards to the cumulative rewards
        self.cumulative_rewards += self.rewards
        
        # mask info tensors with the do_reset tensor
        
        self.masked_num_steps_in_ep = self.do_reset * self.num_steps_in_ep
        
        self.masked_cumulative_rewards = self.do_reset * self.cumulative_rewards
        
        # reset the cumulative rewards for the do_reset values
        self.cumulative_rewards = self.cumulative_rewards * (1- self.do_reset)
        
        # add the masked info tensors to the extras for the step
        self.extras["episode_reward"] = self.masked_cumulative_rewards.to(self.rl_device)
        self.extras["time_outs"] = self.is_timeout.to(self.rl_device)
        self.extras["episode_steps"] = self.masked_num_steps_in_ep
        
        
        self.global_step += 1
        
        if self.rl_device == 'cpu':
            return dict_to_cpu(self.actor_obs), self.rewards.cpu(), self.do_reset.cpu(), self.extras
        return self.actor_obs, self.rewards, self.do_reset, self.extras
    
    def reset(self) -> Tuple[Dict[str, torch.Tensor]]:
        """Reset the complete environment and return a output multispace
        Returns:
            Tuple[Dict[str, torch.Tensor]]: Output multispace (names in the dict correspond to those given in the multispace), (actor_obs, critic_obs)
        """
        actions = torch.zeros([self.num_envs, self.action_space.shape[0] ], dtype=torch.float32, device=self.rl_device)

        # step the simulator
        self.step(actions)
         
        if self.rl_device == 'cpu':
            return dict_to_cpu(self.actor_obs)
        return self.actor_obs

    def close(self) -> None:
        """Close the environment properly
        """
        
    
    @abstractmethod
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different inputs the actor of the agent
         (this includes linear observations, viusal observations)
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference
        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()    
      
    @abstractmethod
    def _get_critic_observation_spaces(self) -> MultiSpace:
        """Define the different inputs for the critic of the agent
        
        This is an asymmetric actor critic implementation  -> The critic observations differ from the actor observations
        and unlike the actor inputs the actor inputs don't have to be things that a real life robot could also observe in inference.
        
        Things like distance to target position, that can not be observed on site can be included in the critic input
    
        Returns:
            MultiSpace: [description]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        raise NotImplementedError()
    
    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(self.actor_observation_spaces.spaces)
            
    @abstractproperty
    def reward_range(self):
        pass
    
    @abstractmethod
    def _extract_params_from_config(self) -> None:
        """Extract important parameters from the config"""
        pass
    
    @abstractmethod
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        
    @abstractmethod
    def _create_envs(self, spacing: float, num_per_row: int)->None:
        pass
    
    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args: 
            actions (torch.Tensor): Expected Shape (num_envs, ) + self._get_action_space.shape
        """

    @abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it.
        
            It is expected, that the implementation of the enviromnment fills the folliwing tensors 
                 - self.do_reset 
                 - self.actor_obs
                 - self.critic_obs
            
        """

    @abstractmethod
    def reset_envs(env_ids: torch.Tensor) -> None:
        """
        Reset the envs of the given env_ids

        Args:
            env_ids (torch.Tensor): A tensor on device, that contains all the ids of the envs that need a reset
            example 
            : tensor([ 0,  10,  22,  43,  51,  64,  81,  82, 99], device='cuda:0')
        """
        pass
        
        