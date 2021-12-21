from typing import Dict, Any, Union, Tuple
from abc import ABC, abstractmethod, abstractproperty

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import numpy as np

import torch 
import torch.nn as nn

import time

import sys

from common.spaces import MultiSpace



        
class Env(ABC):
    
    def __init__(self,config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool) -> None:
        """An asymmetric actor critic base environment class based on isaac gym 

        Args:
            config (Dict[str, Any]): the config dictionary
            sim_device (str): ex: cuda:0, cuda or cpu
            graphics_device_id (int): The device id to render with
            headless (bool): determines whether the scene is rendered
        """
         
                
        self.config = config
        self.headless = headless
        
        
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        
        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", "cuda:0")
        
        
        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:   
            self.graphics_device_id = -1
        
        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        # The Frequency with which the actions are polled relative to physics step
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)
        
        
        # This implementation used Asymetic Actor Critics
        # https://arxiv.org/abs/1710.06542
        self.critic_observation_spaces = self._get_critic_observation_spaces()
        self.actor_observation_spaces = self._get_actor_observation_spaces()
        self.action_space = self._get_action_space()
        
        
        
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info
        """
        
        
        
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the complete environment and return a output multispace
        Returns:
            Dict[str, torch.Tensor]: Output multispace (names in the dict correspond to those given in the multispace),
        """
        pass


    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments
    
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

class VecTask(Env, ABC):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool) -> None:
        """Initialise the `VecTask`.

        Args:
            config (Dict[str, Any]): the config dictionary
            sim_device (str): ex: cuda:0, cuda or cpu
            graphics_device_id (int): The device id to render with
            headless (bool): determines whether the scene is rendered
        """
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        
        self.sim_params = self.__parse_sim_params(self.config["physics_engine"], self.config["sim"])
        if self.config["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.config["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.config['physics_engine']}"
            raise ValueError(msg)
        
        # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)
        
        self.gym = gymapi.acquire_gym()
        
        self.sim_initialized = False
        self.sim = self.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # create the environments 
        print(f'num envs {self.num_envs} env spacing {self.config["env"]["envSpacing"]}')
        self._create_envs( self.config["env"]["envSpacing"], int(np.sqrt(self.num_envs)))
        
        
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True
        
        self.set_viewer()
        
    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
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
    
    def set_viewer(self):
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
        
    def render(self):
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
    
    @abstractmethod
    def _create_envs(self, num_envs: int, spacing: float, num_per_row: int)->None:
        pass
    
    
    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:    
        """Step the physics sim of the environment and apply the given actions

        Args:
            actions (torch.Tensor): [description]

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info
        """
        
        self.pre_physics_step(actions)
        
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            
    
        self.post_physics_step()
    
    
    def _create_ground_plane(self, sim = None):
        print(sim)
        print(sim == None)
        if sim == None:
            sim = self.sim
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        #plane_params.static_friction = self.plane_static_friction
        #plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(sim, plane_params)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
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
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params
