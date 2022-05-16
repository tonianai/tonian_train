from typing import Dict, Any, Union, Tuple
from abc import ABC, abstractmethod

from isaacgym import gymtorch, gymapi
from tonian.tasks.base.base_env import BaseEnv
from tonian.common.utils import  join_configs

from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import numpy as np

import torch, sys, yaml, os


class VecTask(BaseEnv, ABC):
    """
    Vectorized Environments are a method for stacking multiple independent environments into a single environment.
    Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step


    """
    
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str)  -> None:
        """Initialise the `VecTask`.

        Args:
            config (Dict[str, Any]): the config dictionary
            sim_device (str): ex: cuda:0, cuda or cpu
            graphics_device_id (int): The device id to render with
            headless (bool): determines whether the scene is rendered
        """
        
        # add the top level configs
        vec_env_base_config = self._get_vec_env_base_config()
         
        config = join_configs(vec_env_base_config, config)
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        # The sum of all action dimensions        
        self._action_size = None
        
        self.num_envs = config["vec_task"]["num_envs"] 

        # The Frequency with which the actions are polled relative to physics step
        self.control_freq_inv = config["vec_task"].get("control_freq_invariant", 1)
 
        
        self.device = "cpu"
        if config["vec_task"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["vec_task"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", rl_device)
        
        
        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:   
            self.graphics_device_id = -1
        
        self._first_domain_randomization= True # determines whether there has alreadly beem an randomization
        self.original_props = {} # properties before randomization
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None
        
        self._extract_params_from_config()
        
        self.sim_params = self.__parse_sim_params(self.config['vec_task']["physics_engine"], self.config['vec_task']["sim"])
        if self.config["vec_task"]["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.config["vec_task"]["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.config['vec_task']['physics_engine']}"
            raise ValueError(msg)
        
        # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)
         
        
        self.gym = gymapi.acquire_gym()
         
        self.global_step = 0
        self.sim_initialized = False
        self.sim = self.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
         
        # create the environments 
        print(f'num envs {self.num_envs} env spacing {self.config["vec_task"]["env_spacing"]}')
        self._create_envs( self.config["vec_task"]["env_spacing"], int(np.sqrt(self.num_envs)))
        
        self.max_episode_length = self.config['vec_task'].get('max_episode_length', 10000)
        
        print(f"The max ep length is {self.max_episode_length}")
        
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.allocate_buffers()
        self.set_viewer()
        
        
        # These are the extras that will be returned on the step function
        self.extras = {}
    
    def get_number_of_agents(self):
        return self.num_envs
             

    def allocate_buffers(self):
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

        
        self.critic_obs = {key: torch.zeros((self.num_envs, ) + space_shape, device=self.device, dtype=torch.float32) for (key,space_shape) in self.critic_observation_spaces.dict_shape.items()}
      

        # Randomize buffer determines whether a given env should randomize 
        self.do_randomize = torch.zeros((self.num_envs, ), device=self.device, dtype= torch.int8 )
        
        # rewards tensor stores the rewards that were given at last
        self.rewards = torch.zeros((self.num_envs, ) , device= self.device, dtype=torch.float32)
        
        # mean values, that make up the reward -> only used for logging 
        self.reward_constituents = {}


        # the cumulative rewards achieved within the environment until this steop
        self.cumulative_rewards = torch.zeros(self.num_envs, device= self.device, dtype= torch.float32)
        
        # the cumulative rewards, but only the envs in terminal states are not hidden
        self.masked_cumulative_rewards = torch.zeros(self.num_envs, device= self.device, dtype= torch.float32)
        
        self.former_actions = torch.zeros((self.num_envs, ) + self.action_space.shape ).to(self.device)
        
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
    
    def _extract_params_from_config(self) -> None:
        """Extract important parameters from the config given in the constructor
        this will be called hierachically"""
        
    def _get_vec_env_base_config(self) -> Dict:
        """Get the base config for the vec_task

        Returns:
            Dict: _description_
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_vec_task.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
     
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
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """
        Reset the envs of the given env_ids

        Args:
            env_ids (torch.Tensor): A tensor on device, that contains all the ids of the envs that need a reset
                example 
                : tensor([ 0,  10,  22,  43,  51,  64,  81,  82, 99], device='cuda:0')
 
        """
        pass
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, int]]:    
        """Step the physics sim of the environment and apply the given actions

        Args: 
            actions (torch.Tensor): Expected Shape (num_envs, ) + self._get_action_space.shape

        Returns:
            Tuple[ Dict[str, torch.Tensor],  torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, float]]: 
            Observations(names in the dict correspond to those given in the multispace), rewards, resets, info, avg_reward_constituents
        """
        
        self.extras = {}
        
        self.pre_physics_step(actions)
        
        for i in range(self.control_freq_inv):
            # simulation loop
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
        self.former_actions = actions.detach().clone()
        
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
        
        
        return (self.actor_obs, self.critic_obs), self.rewards, self.do_reset, self.extras, self.reward_constituents
        
    def reset(self) -> Tuple[Dict[str, torch.Tensor]]:
        """Reset the environment 

        Returns:
            Tuple[Dict[str, torch.Tensor]]: actor_obs, critic_ob
        """
         
        actions = torch.zeros([self.num_envs, self.action_space.shape[0] ], dtype=torch.float32, device=self.rl_device)
         
        every_env_id_tensor = torch.arange(start=0, end= self.num_envs).to(self.device).to(dtype=torch.int64)
        
        self.initial_domain_randomization()
        self.reset_envs(every_env_id_tensor )
        
        
        # step the simulator
        self.step(actions)
        
        return self.actor_obs, self.critic_obs 
 
    def _create_ground_plane(self, sim = None):
        if not sim:
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
    
    @abstractmethod
    def get_num_actors_per_env(self) -> int:
        """Return the amount of actors each environment has, this includes actors, that are not playable
        This distincion is stupid and only exists, because isaacgym currently does anot support any way of adding objects to environments, that are not actors

        Returns:
            int
        """
        raise NotImplementedError()
    
    
    def get_num_playable_actors_per_env(self) -> int:
        """Return the amount of actors each environment has, this only includes actors, that are playable
        This distincion is stupid and only exists, because isaacgym currently does anot support any way of adding objects to environments, that are not actors

        Returns:
            int
        """
        return self.get_num_actors_per_env()
    
    
    @property
    def action_size(self):
        """The sum over all the action dimension 
        """
        if not self._action_size:
             self._action_size =  self.action_space.sample().reshape(-1).shape[0]
        
        return self._action_size
             
    def apply_domain_randomization(self, env_ids: torch.Tensor):
        """Apply domain randomisation to the parameters given in the config file
        
        This Function should be called by subclasses on env reset, either by using the super() or by calling directly
        Args:
            env_ids (torch.Tensor): ids where dr should be performed (typically the env_ids, that are resetting) 
        """
        
    def initial_domain_randomization(self):
        """Apply domain randomization to the parameter given in the config file after reset
        
        This Function should be called by subclasses on env reset, either by using the super() or by calling directly
        
        """
        pass
        
        
             

 