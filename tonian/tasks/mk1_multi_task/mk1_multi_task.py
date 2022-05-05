
from abc import ABC, abstractmethod

from gym.spaces import space
import numpy as np
from tonian.tasks.common.command import Command
from tonian.tasks.base.vec_task import VecTask
from tonian.tasks.common.task_dists import task_dist_from_config
from tonian.common.utils import join_configs

from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace

from tonian.common.torch_jit_utils import batch_dot_product
 

from typing import Dict, Any, Tuple, Union, Optional, List

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch

import os, torch, gym, yaml, time
 
class Mk1Multitask(VecTask):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:

        # all the link names
        self.link_names = ['upper_torso' , 'torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]

        # The parts of the robot, that should get a force sensor    
        self.parts_with_force_sensor = ['upper_torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]
        
        base_config = self._get_standard_config()
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
        
        self.default_friction = task_dist_from_config(mk1_config.get('default_friction', 1.0))
        self.friction_properties = { key: task_dist_from_config(friction) for key, friction in mk1_config.get('frictions', {}).items()}
        
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
        
        print(self.initial_root_states.shape)
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
        
    
    
    def reset_envs(self, env_ids: torch.Tensor, do_reset_bool_tensor: torch.Tensor) -> None:
        """
        Reset the envs of the given env_ids

        Args:
            env_ids (torch.Tensor): A tensor on device, that contains all the ids of the envs that need a reset
                example 
                : tensor([ 0,  10,  22,  43,  51,  64,  81,  82, 99], device='cuda:0')

            do_reset_bool_tensor (torch.Tensor): Carries the same information as the env_ids tensor, but encoded as a boolean tensor at the position indices that do reset with a 1 and env positions that do not reset with a zero
                example
                : tensor([0,0,0,0,0,1,0,0,1,0,0,0,0,0], device='cuda:0')
        """
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)    
        
        
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        self.apply_domain_randomization(env_ids=env_ids_int32, do_reset_bool_tensor= do_reset_bool_tensor)
        
         
 
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
        num_actor_obs = 108
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

        num_critic_obs = 6
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
    
            
    def initial_domain_randomization(self):
        """Initial domain randomization across all the envs
            also saves tensors 
        """
        # ---- actor linear velocity domain randomization ----
        # update the initial root state linear velocities with the randomized values given by the task std
        self.initial_root_states[: , 7] = self.intitial_velocities[0]()  # vel in -y axis
        self.initial_root_states[:, 8] = self.intitial_velocities[1]() # vel in -x axis
        self.initial_root_states[:, 9] = self.intitial_velocities[2]() # vel in -z axis
        
        
            
    def apply_domain_randomization(self, env_ids: torch.Tensor, do_reset_bool_tensor: torch.Tensor):
        """Apply domain randomisation to the parameters given in the config file
        
        This Function should be called by subclasses on env reset, either by using the super() or by calling directly
        Args:
            env_ids (torch.Tensor): ids where dr should be performed (typically the env_ids, that are resetting)
            do_reset_bool_tensor (torch.Tensor): Carries the same information as the env_ids tensor, but encoded as a boolean tensor at the position indices that do reset with a 1 and env positions that do not reset with a zero
       
        """
        super().apply_domain_randomization(env_ids, do_reset_bool_tensor)
        
        # ---- actor linear velocity domain randomization ----
        
        # update the initial root state linear velocities with the randomized values given by the task std
        self.initial_root_states[do_reset_bool_tensor , 7] = self.intitial_velocities[0]()  # vel in -y axis
        self.initial_root_states[do_reset_bool_tensor , 8] = self.intitial_velocities[1]() # vel in -x axis
        self.initial_root_states[do_reset_bool_tensor , 9] = self.intitial_velocities[2]() # vel in -z axis
        
        shape_properties = self.gym.get_actor_rigid_shape_properties(self.envs[0], self.robot_handles[0])
        
        
        for env_id in env_ids: # this is very costly 
            
            
            # ---- friction domain randomization -----
            shape_properties = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.robot_handles[env_id])
            
            friction_values = [ self.default_friction() for _ in range(len(shape_properties))]
            
            for link_name, friction_value in self.friction_properties:
                link_index = self.gym.find_asset_rigid_body_index(self.mk1_robot_asset,link_name)
                friction_values[link_index] = friction_value()
                
            for shape_property, friction_value in zip(shape_properties,  friction_values):
                shape_property.friction = friction_value
                
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.robot_handles[env_id], shape_properties)
                
            
            # ---- mass domain randomization ....
            body_properties = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id])
            
            mass_values = [ self.default_friction() for _ in range(len(shape_properties))]

                
                
        
            # print(self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id]))
            
            # object_properties = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id])
            # 
            # for property in object_properties:
            #     print(property.mass)
            # 
            # for property in object_properties:
            #     property.mass = 1000
            # 
            # self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.robot_handles[env_id], object_properties)
            
            
            #for property in shape_properties:
            #    print(property.friction)
            pass
    
    def get_num_actors_per_env(self) -> int:
        return 1
    
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
         
        assert self.config["mk1_multitask"] is not None, "The mk1_multitask config must be set on the task config file"
        
        reward_weight_dict = self.config["mk1_multitask"]["reward_weighting"]  
        
        self.energy_cost = reward_weight_dict["energy_cost"]
        self.forward_directional_factor = reward_weight_dict["forward_directional_factor"]
        self.death_cost = reward_weight_dict["death_cost"]
        self.alive_reward = float(reward_weight_dict["alive_reward"])
        self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        self.death_height = reward_weight_dict["death_height"]
        self.overextend_cost = reward_weight_dict["overextend_cost"]
        self.die_on_contact = reward_weight_dict.get("die_on_contact", True)
        self.contact_punishment_factor = reward_weight_dict["contact_punishment"]

    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor,]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
               Tuple[torch.Tensor, torch.Tensor]: 
                   reward: torch.Tensor shape(num_envs, )
                   has_fallen: torch.Tensor shape(num_envs, )
                   constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
        """
    
    
        # -------------- base reward for being alive --------------  
        
        reward = torch.ones_like(self.root_states[:, 0]) * self.alive_reward  
        
        
        quat_rotation = self.root_states[: , 3:7]
        
        #  -------------- reward for an upright torso -------------- 
        
        # The upright value ranges from 0 to 1, where 0 is completely horizontal and 1 is completely upright
        # Calulation explanation: 
        # take the first and the last value of the quaternion and take the quclidean distance
        upright_value = torch.sqrt(torch.sum( torch.square(quat_rotation[:, 0:4:3]), dim= 1 ))
        
        upright_punishment = (upright_value -1) * self.upright_punishment_factor
        
        reward += upright_punishment
        
        #  -------------- reward for speed in the forward direction -------------- 
        # Note; When we want the robot to go to a taret, it should not go backwards to that target, but actually turn and then walk forward in the direction of that target
        
        linear_velocity_x_y = self.root_states[: , 7:9]
        
        # direction_in_deg base is -> neg x Axis
        direction_in_deg_to_x = torch.acos(quat_rotation[:, 0]) * 2
        
        # unit vecor of heading when seen from above 
        # this unit vector makes little sense, when the robot is highly non vertical
        two_d_heading_direction = torch.transpose(torch.cat((torch.unsqueeze(torch.sin(direction_in_deg_to_x), dim=0), torch.unsqueeze(torch.cos(direction_in_deg_to_x),dim=0) ), dim = 0), 0, 1)
        
        # compare the two_d_heading_direction with the linear_velocity_x_y using the angle between them
        # magnitude of the velocity (2 norm)
        vel_norm = torch.linalg.vector_norm(linear_velocity_x_y, dim=1)
        
            
        #heading_to_velocity_angle = torch.arccos( torch.dot(two_d_heading_direction, linear_velocity_x_y)  / vel_norm )
        heading_to_velocity_angle = torch.arccos( batch_dot_product(two_d_heading_direction, linear_velocity_x_y) / vel_norm)
        
        forward_direction_reward = torch.where(torch.logical_and(upright_value > 0.7, heading_to_velocity_angle < 0.5), vel_norm * self.forward_directional_factor, torch.zeros_like(reward))
    
        reward += forward_direction_reward  
        
        # -------------- Punish for jittery motion (see ./research/2022-03-27_reduction-of-jittery-motion-in-action.md)--------------
        
        jitter_punishment = torch.abs(self.actions - self.former_actions).view(reward.shape[0], -1).sum(-1) * self.jitter_cost
        reward -= jitter_punishment
        
        
        
        #-------------- cost for overextension --------------
        distance_to_upper = self.dof_limits_upper - self.dof_pos
        distance_to_lower = self.dof_pos - self.dof_limits_lower
        distance_to_limit = torch.minimum(distance_to_upper, distance_to_lower)
        
        # 0.001 rad -> 0,071 deg 
        at_upper_limit = torch.where(distance_to_upper < 0.02, self.actions, torch.zeros_like(distance_to_limit))
        at_lower_limit = torch.where(distance_to_lower < 0.02, self.actions, torch.zeros_like(distance_to_lower)) * -1
        at_lower_limit[:, 8] = 0
        at_upper_limit[: , 8] = 0
        
        clipped_upper_punishment = torch.clamp(at_upper_limit, min=0) * self.overextend_cost
        clipped_lower_punishment = torch.clamp(at_lower_limit, min=0) * self.overextend_cost
        
        overextend_punishment = torch.sum(clipped_lower_punishment + clipped_upper_punishment, dim=1) / clipped_lower_punishment.shape[1]
        
        reward -= overextend_punishment
        
    
        
        
        # -------------- cost of power --------------
        
        energy_punishment = torch.sum(self.actions ** 2, dim=-1) * self.energy_cost
        reward -= energy_punishment
         
        terminations_height = self.death_height
        
        
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(self.root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
        
        
        
        summed_contact_forces = torch.sum(self.contact_forces, dim= 2) # sums x y and z components of contact forces together
        
        summed_contact_forces[:,self.left_foot_index] = 0.0
        summed_contact_forces[:, self.right_foot_index] = 0.0
        
        total_summed_contact_forces = torch.sum(summed_contact_forces, dim=1) # sum all the contact forces of the other indices together, to see if there is any other contact other than the feet
        
        has_contact = torch.where(total_summed_contact_forces > torch.zeros_like(total_summed_contact_forces), torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        if self.die_on_contact:
            has_fallen += has_contact
        else:
            n_times_contact = (summed_contact_forces > 0 ).to(dtype=torch.float32).sum(dim=1)
            
            contact_punishment = n_times_contact * self.contact_punishment_factor
            
            reward -= contact_punishment
        
        # ------------- cost for dying ----------
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(has_fallen == 1, - 1 * torch.ones_like(reward) * self.death_cost, reward)
        
    
        
        # average rewards per step
         
        upright_punishment = float(torch.mean(upright_punishment).item())
        forward_direction_reward = float(torch.mean(forward_direction_reward).item())
        jitter_punishment = - float(torch.mean(jitter_punishment).item())
        energy_punishment = - float(torch.mean(energy_punishment).item())
        overextend_punishment = - float(torch.mean(overextend_punishment).item())
        if not self.die_on_contact:
            contact_punishment = -float(torch.mean(contact_punishment).item())
        else:
            contact_punishment = 0.0
        
        total_avg_reward = self.alive_reward + upright_punishment + forward_direction_reward + jitter_punishment + energy_punishment
        
        reward_constituents = {
                                'alive_reward': self.alive_reward,
                                'upright_punishment':  upright_punishment,
                                'forward_direction_reward':    forward_direction_reward,
                                'jitter_punishment':   jitter_punishment,
                                'energy_punishment':   energy_punishment,
                                'overextend_punishment': overextend_punishment,
                                'contact_punishment': contact_punishment,
                                'total_reward': total_avg_reward
                            }
        
        
        return (reward, has_fallen, reward_constituents)
            
    
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_mk1_multi.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            

    
        

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
    
    
    linear_actor_obs = torch.cat((sensor_states.view(root_states.shape[0], -1), dof_pos, dof_vel, dof_force, ang_velocity, torso_rotation, actions, torso_position), dim=-1)
    
    linear_critic_obs = torch.cat(( velocity, torso_position), dim=-1)
    
    return  linear_actor_obs,   linear_critic_obs



