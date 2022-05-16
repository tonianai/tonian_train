
from abc import ABC, abstractmethod

from gym.spaces import space
import numpy as np 
from tonian.tasks.base.vec_task import VecTask
from tonian.tasks.common.task_dists import task_dist_from_config, TaskGaussianDistribution
from tonian.common.utils import join_configs


from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace

 

from typing import Dict, Any, Tuple, Union, Optional, List

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import os, torch, gym, yaml, time
 
class Mk1BaseClass(VecTask, ABC):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:

        # all the link names
        self.link_names = ['upper_torso' , 'torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]

        # The parts of the robot, that should get a force sensor    
        self.parts_with_force_sensor = ['upper_torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]
        
        
        
        base_config = self._get_mk1_base_config()
        config = join_configs(base_config, config)
        
        self._parse_config_params(config)
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        self.action_space_shape = self.action_space.sample().shape
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
    def _parse_config_params(self, config):
        """
        Parse the mk1 values from the config to be used in the implementation
        """
        
        mk1_config = config['mk1']
        
        # The initial velcoities after the environment resets
        self.intitial_velocities = [  task_dist_from_config(vel) for vel in mk1_config.get('initial_velocities', [0,0,0])]
        
        
        
        self.spawn_height = mk1_config.get('spawn_height', 1.7)
        
        self.default_friction = task_dist_from_config(mk1_config['agent'].get('default_friction', 1.0))
        self.friction_properties = { key: task_dist_from_config(friction) for key, friction in mk1_config['agent'].get('frictions', {}).items()}
        
        # the standard deviation with wicthc to randomize the mass of each part of the robot
        self.mass_std = mk1_config['agent'].get('default_mass_std', 0)
        
    def _get_mk1_base_config(self):
        """Get the base config for the vec_task

        Returns:
            Dict: _description_
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_mk1_base.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            
    def _get_gpu_gym_state_tensors(self) -> None:
        """
        Retreive references to the gym tensors for the enviroment, that are on the gpu
        """
        # --- aquire tensor pointers
        # the state of each root body is represented using 13 floats with the same layout as GymRigidBodyState: 3 floats for position, 4 floats for quaternion, 3 floats for linear velocity, and 3 floats for angular velocity.
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # Retrieves buffer for Actor root states. Buffer has shape (num_environments, num_actors * 13).
        # State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # Retrieves Degree-of-Freedom state buffer. Buffer has shape (num_environments, num_dofs * 2).
        # Each DOF state contains position and velocity.
        
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) 
        # Retrieves buffer for DOF forces. One force value per each DOF in simulation.
        # shape (num_envs * dofs, ) WHYYY??? 
        
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # Retrieves buffer for force sensors. Buffer has shape (num_sensors,  6). 
        # Each force sensor state has forces (3) and torques (3) data.
        
        
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        # Retrieves buffer for net contract forces. Buffer has shape (num_environments, num_bodies * 3). 
        # Each contact force state contains one value for each X, Y, Z axis.
      
        # --- wrap pointers to torch tensors (The iaacgym simulation tensors must be wrapped to get a torch.Tensor)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        # view is necesary, because of the linear shape provided by the self.gym.acuire_dof_force_tensor
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        # the amount of sensors each env has
        sensors_per_env = len(self.parts_with_force_sensor)
        self.vec_force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env, 6)
        
        self.refresh_tensors()
        
        # positions of the joints
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # velocities of the joints
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # --- set initial tensors
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        self.initial_root_states = self.root_states.clone()
        
        # 7:13 describe velocities
        self.initial_root_states[:, 7:13] = 0
        
         
        self.initial_root_states[: , 7] = self.intitial_velocities[0]()  # vel in -y axis
        self.initial_root_states[: , 8] = self.intitial_velocities[1]() # vel in -x axis
        self.initial_root_states[: , 9] = self.intitial_velocities[2]() # vel in -z axis
        
    def refresh_tensors(self):
        """Refreshes tensors, that are on the GPU
        """
        # The root state that was from last refresh
        self.former_root_states = self.root_states.clone()
        self.gym.refresh_dof_state_tensor(self.sim) # state tensor contains velocities and position for the jonts 
        self.gym.refresh_actor_root_state_tensor(self.sim) # root state tensor contains ground truth information about the root link of the actor
        self.gym.refresh_force_sensor_tensor(self.sim) # the tensor of the added force sensors (added in _create_envs)
        self.gym.refresh_dof_force_tensor(self.sim) # dof force tensor contains foces applied to the joints
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        
    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        """Create all the environments and initialize the agens in those environments

        Args:
            spacing (float): _description_
            num_per_row (int): _description_
        """
        
        
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)


        self.mk1_robot_asset = self.create_mk1_asset(self.config['mk1']['pure_shapes'])

        
        self.num_dof = self.gym.get_asset_dof_count(self.mk1_robot_asset) 
                
                
        self.robot_handles = []
        self.envs = [] 
        
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0,0.0, self.spawn_height)
        start_pose.r = gymapi.Quat(0.0, 0.0 , 0.0, 1.0)
        
        self._motor_efforts = self._create_effort_tensor(self.mk1_robot_asset)
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(
                env= env_ptr, 
                asset = self.mk1_robot_asset,
                pose = start_pose,
                name = "mk1",
                group = i, 
                filter = 1,
                segmentationId = 0)
            
             
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            
            
            self._add_to_env(env_ptr, i )
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            
            
        # get all dofs and assign the action index to the dof name in the dof_name_index_dict
        self.dof_name_index_dict = self.gym.get_actor_dof_dict(env_ptr, robot_handle)
        
        self.left_foot_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], 'foot')
        self.right_foot_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], 'foot_2')
        
        # take the last one as an example (All should be the same)
        dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
        
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        
        
    def _create_effort_tensor(self, mk1_robot_asset) -> torch.Tensor:
        """Create the motor effort tensor, that defines how strong each motor is 

        Returns:
            torch.Tensor: shape(self.action_size)
        """
         
        
        if 'agent' not in self.config['mk1']:
            default_motor_effort = 1000
            return  (torch.ones(self.action_size) * default_motor_effort).to(self.device)
        else:
            default_motor_effort =  self.config['mk1']['agent'].get('default_motor_effort', 1000)
            motor_efforts = (torch.ones(self.action_size) * default_motor_effort).to(self.device) 
            
            # specific motor efforts
            specific_motor_efforts = self.config['mk1']['agent'].get('motor_powers', {})
            
            for motor_name, motor_effort in specific_motor_efforts.items():
                
                dof_id = self.gym.find_asset_dof_index(mk1_robot_asset, motor_name)
                
                motor_efforts[dof_id] = motor_effort
                
            
            return motor_efforts
    
    def create_mk1_asset(self, use_pure_shapes: bool = False):
        """Create the Mk1 Robot with all sensors and setting in the simulation and return the asset

        Args:
            gym (_type_): isaacgym gym reference
            sim (_type_): isaacgym sim reference
        """

        if use_pure_shapes: 
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/mk-1/mk1-pure-shapes")
        else: 
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/mk-1/mk1-precision")

        mk1_robot_file = "robot.urdf"

        asset_options = gymapi.AssetOptions()


        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT # DOM_MODE_EFFORT and DOF_MODE_NONE seem to be similar
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False


        mk1_robot_asset = self.gym.load_asset(self.sim, asset_root, mk1_robot_file, asset_options)
 
        sensor_pose = gymapi.Transform()
 

        parts_idx = [self.gym.find_asset_rigid_body_index(mk1_robot_asset, part_name) for part_name in self.parts_with_force_sensor]
 

        for part_idx in parts_idx:
            self.gym.create_asset_force_sensor(mk1_robot_asset, part_idx, sensor_pose)

        return mk1_robot_asset
    
        
    
    @abstractmethod
    def _add_to_env(self, env_ptr):
        """During the _create_envs this is called to give mk1_envs the ability to add additional things to the environment

        Args:
            env_ptr (_type_): pointer to the env
        """
        pass
        
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the action given to all the envs
        Args:
            actions (torch.Tensor): Expected Shape (num_envs, ) + self._get_action_space.shape

        Returns:
            [type]: [description]
        """
         
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self._motor_efforts
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
    
    def post_physics_step(self):
        """Compute observations and calculate Reward
        """
        # the data in the tensors must be correct, before the observation and reward can be computed
        self.refresh_tensors()
        
        # compute the observations using the implementation in the subclass
        self.get_obs()
        
         
        self.rewards , self.do_reset , self.reward_constituents = self._compute_robot_rewards()
        
    @abstractmethod
    def get_obs(self):
        """Compute all of the observations and return both the actor and the critic obstervations
        
        -> set the self.actor_obs and the self.critic_obs dicts

        """
        pass
    
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """
        Reset the envs of the given env_ids

        Args:
            env_ids (torch.Tensor): A tensor on device, that contains all the ids of the envs that need a reset
                example 
                : tensor([ 0,  10,  22,  43,  51,  64,  81,  82, 99], device='cuda:0')
 
        """
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)    
        
        
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        
        
        self.apply_domain_randomization(env_ids=env_ids)
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        
        
         
 
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                   gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
  
    def _get_action_space(self) -> gym.Space:
        """The action space is only a single gym space and most often a suspace of the multispace output_space 
        Returns:
            gym.Space: [description]
        """
        num_actions = 17
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions, )) 
    
        
    def reward_range(self):
        return (-1e100, 1e100)
    
    def close(self):
        pass
    
    @abstractmethod
    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
                      Tuple[torch.Tensor, torch.Tensor]: 
                          reward: torch.Tensor shape(num_envs, )
                          has_fallen: torch.Tensor shape(num_envs, )
                          constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
      
        """
        raise NotImplementedError()
    
            
    def initial_domain_randomization(self):
        """Initial domain randomization across all the envs
            also saves tensors 
        """
        # ---- actor linear velocity domain randomization ----
        # update the initial root state linear velocities with the randomized values given by the task std
        # self.initial_root_states[: , 7] = self.intitial_velocities[0]()  # vel in -y axis
        # self.initial_root_states[:, 8] = self.intitial_velocities[1]() # vel in -x axis
        # self.initial_root_states[:, 9] = self.intitial_velocities[2]() # vel in -z axis
        
        
        self.mass_add_dist = TaskGaussianDistribution({'mean':1.0, 'std': self.mass_std })
        self.default_mass_values = [prop.mass for prop in self.gym.get_actor_rigid_body_properties(self.envs[0], self.robot_handles[0])]
        
            
    def apply_domain_randomization(self, env_ids: torch.Tensor):
        """Apply domain randomisation to the parameters given in the config file
        
        This Function should be called by subclasses on env reset, either by using the super() or by calling directly
        Args:
            env_ids (torch.Tensor): ids where dr should be performed (typically the env_ids, that are resetting) 
    
        """
        super().apply_domain_randomization(env_ids)
        
        
        # ---- actor linear velocity domain randomization ----
        
        env_ids = env_ids.to(torch.int64)
        
        # update the initial root state linear velocities with the randomized values given by the task std
        self.initial_root_states[env_ids , 7] = self.intitial_velocities[0]()  # vel in -y axis
        self.initial_root_states[env_ids , 8] = self.intitial_velocities[1]() # vel in -x axis
        self.initial_root_states[env_ids , 9] = self.intitial_velocities[2]() # vel in -z axis
        
        shape_properties = self.gym.get_actor_rigid_shape_properties(self.envs[0], self.robot_handles[0])
        
        
        for env_id in env_ids: # this is very costly 
            
            
            # ---- friction domain randomization -----
            shape_properties = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.robot_handles[env_id])
            
            friction_values = [ self.default_friction() for _ in range(len(shape_properties))]
            
            for link_name, friction_value in self.friction_properties.items():
                link_index = self.gym.find_asset_rigid_body_index(self.mk1_robot_asset,link_name)
                friction_values[link_index] = friction_value()
                
            for shape_property, friction_value in zip(shape_properties,  friction_values):
                shape_property.friction = friction_value  
                
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.robot_handles[env_id], shape_properties)
                
            
            # ---- mass domain randomization ....
            body_properties = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id])
            
            mass_values = self.default_mass_values.copy()
            
            
            for i, property in enumerate(body_properties):
                 property.mass =  mass_values[i] * self.mass_add_dist.sample()
                 
                
            
            self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id], body_properties)
            

            # print(self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id]))
            
            # object_properties = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id])
            # 
            # for property in object_properties:
            #     print(property.mass)
            # 
            # for property in object_properties:
            #     property.mass = 1000
            # 
            
            
            #for property in shape_properties:
            #    print(property.friction)
            pass
      
    
    
        
