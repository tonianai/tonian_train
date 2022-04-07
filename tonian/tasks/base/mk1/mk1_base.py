
from abc import ABC, abstractmethod

from gym.spaces import space
import numpy as np
from tonian.tasks.base.command import Command
from tonian.tasks.base.vec_task import VecTask

from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace

import gym

from typing import Dict, Any, Tuple, Union, Optional, List

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import yaml, os, torch


class Mk1BaseClass(VecTask, ABC):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:

        # The parts of the robot, that should get a force sensor    
        self.parts_with_force_sensor = ['upper_torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]
        
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        self.action_space_shape = self.action_space.sample().shape
        

        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
    
            
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
        
      
        # --- wrap pointers to torch tensors (The iaacgym simulation tensors must be wrapped to get a torch.Tensor)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
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
        
        initial_velocities = self.config['env'].get("initial_velocities", [0 ,0,0])
        self.initial_root_states[: , 7] = initial_velocities[0] # vel in -y axis
        self.initial_root_states[: , 8] = initial_velocities[1] # vel in -x axis
        self.initial_root_states[: , 9] = initial_velocities[2] # vel in -z axis
        
        
    def refresh_tensors(self):
        """Refreshes tensors, that are on the GPU
        """
        # The root state that was from last refresh
        self.former_root_states = self.root_states.clone()
        self.gym.refresh_dof_state_tensor(self.sim) # state tensor contains velocities and position for the jonts 
        self.gym.refresh_actor_root_state_tensor(self.sim) # root state tensor contains ground truth information about the root link of the actor
        self.gym.refresh_force_sensor_tensor(self.sim) # the tensor of the added force sensors (added in _create_envs)
        self.gym.refresh_dof_force_tensor(self.sim) # dof force tensor contains foces applied to the joints
    
    def _create_envs(self, spacing: float, num_per_row: int) -> None:
        """Create all the environments and initialize the agens in those environments

        Args:
            spacing (float): _description_
            num_per_row (int): _description_
        """
        
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)


        mk1_robot_asset = self.create_mk1_asset()

        
        self.num_dof = self.gym.get_asset_dof_count(mk1_robot_asset) 
                
                
        self.robot_handles = []
        self.envs = [] 
        
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0,0.0, 1.80)
        start_pose.r = gymapi.Quat(0.0, 0.0 , 0.0, 1.0)
        
        self._motor_efforts = self._create_effort_tensor(mk1_robot_asset)
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, mk1_robot_asset, start_pose, "mk1", i, 1, 0)
            
             
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            
            
            self._add_to_env(env_ptr)
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            
            
        # get all dofs and assign the action index to the dof name in the dof_name_index_dict
        self.dof_name_index_dict = self.gym.get_actor_dof_dict(env_ptr, robot_handle)
        
        
        
        
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
         
        
        if 'agent' not in self.config['env']:
            default_motor_effort = 1000
            return  (torch.ones(self.action_size) * default_motor_effort).to(self.device)
        else:
            default_motor_effort =  self.config['env']['agent'].get('default_motor_effort', 1000)
            motor_efforts = (torch.ones(self.action_size) * default_motor_effort).to(self.device) 
            
            # specific motor efforts
            specific_motor_efforts = self.config['env']['agent'].get('motor_powers', {})
            
            
            
            for motor_name, motor_effort in specific_motor_efforts.items():
                
                dof_id = self.gym.find_asset_dof_index(mk1_robot_asset, motor_name)
                
                motor_efforts[dof_id] = motor_effort
                
            
            return motor_efforts
    
    def create_mk1_asset(self):
        """Create the Mk1 Robot with all sensors and setting in the simulation and return the asset

        Args:
            gym (_type_): isaacgym gym reference
            sim (_type_): isaacgym sim reference
        """


        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/urdf/mk-1/")

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
        
         # use jit script to compute the observations
        self.actor_obs["linear"][:], self.critic_obs["linear"][:] = compute_linear_robot_observations(
            root_states = self.root_states, 
            sensor_states=self.vec_force_sensor_tensor,
            dof_vel=self.dof_vel,
            dof_pos= self.dof_pos,
            dof_limits_lower=self.dof_limits_lower,
            dof_limits_upper=self.dof_limits_upper,
            dof_force= self.dof_force_tensor, 
            actions= self.actions
        )
         
        self.rewards , self.do_reset , self.reward_constituents = self._compute_robot_rewards()
        
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)    
        
        
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        
 
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                   gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_actor_obs = 105
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, ))
        })
        
    def _get_critic_observation_spaces(self) -> MultiSpace:
        """Define the different observations for the critic of the agent
        
        
         The observations will later be combined with other inputs like commands to create the critic input space
        
        This is an asymmetric actor critic implementation  -> The critic observations differ from the actor observations
        and unlike the actor inputs the actor inputs don't have to be things that a real life robot could also observe in inference.
        
        Things like distance to target position, that can not be observed on site can be included in the critic input
    
        Returns:
            MultiSpace: [description]
        """

        num_critic_obs = 132
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_critic_obs, ))
        })
    
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
    
    
    def _joint_name_to_action_index(self, joint_names: Union[str, List[str]]) -> torch.Tensor:
        """Get the action indices of the given joint names

        Args:
            joint_names (Union[str, List[str]]): _description_

        Returns:
            torch.Tensor: _description_
        """
        if isinstance(joint_names, str):
            joint_names = [joint_names]
            
        




@torch.jit.script
def compute_linear_robot_observations(root_states: torch.Tensor, 
                                sensor_states: torch.Tensor, 
                                dof_vel: torch.Tensor, 
                                dof_pos: torch.Tensor, 
                                dof_limits_lower: torch.Tensor,
                                dof_limits_upper: torch.Tensor,
                                dof_force: torch.Tensor,
                                actions: torch.Tensor
                                ):
    
    
    """Calculate the observation tensors for the crititc and the actor for the humanoid robot
    
    Note: The resulting tensors must be in the same shape as the multispaces: 
        - self.actor_observation_spaces
        - self.critic_observatiom_spaces

    Args:
        root_states (torch.Tensor): Root states contain things like positions, velcocities, angular velocities and orientation of the root of the robot 
        sensor_states (torch.Tensor): state of the sensors given 
        dof_vel (torch.Tensor): velocity tensor of the dofs
        dof_pos (torch.Tensor): position tensor of the dofs
        dof_force (torch.Tensor): force tensor of the dofs
        actions (torch.Tensor): actions of the previous 

    Returns:
        Tuple[Dict[torch.Tensor]]: (actor observation tensor, critic observation tensor)
    """
    
    
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
     
    
    
    # todo add some other code to deal with initial information, that might be required
    
    
    linear_actor_obs = torch.cat((sensor_states.view(root_states.shape[0], -1), dof_pos, dof_vel, dof_force, ang_velocity, torso_rotation, actions), dim=-1)
    
    linear_critic_obs = torch.cat((linear_actor_obs, torso_rotation, velocity, torso_position, actions), dim=-1)
    
    return  linear_actor_obs,   linear_critic_obs



