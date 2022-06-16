
from abc import ABC, abstractmethod

from gym.spaces import space
import numpy as np 
from tonian.tasks.base.vec_task import VecTask
from tonian.tasks.mk1.mk1_base import Mk1BaseClass
from tonian.tasks.common.task_dists import task_dist_from_config
from tonian.common.utils import join_configs

from isaacgym.torch_utils import torch_rand_float, tensor_clamp

from tonian.common.spaces import MultiSpace
from tonian.common.torch_jit_utils import batch_dot_product, batch_normalize_vector, get_batch_tensor_2_norm
from tonian.tasks.common.terrain import Terrain

from tonian.tasks.common.task_dists import sample_tensor_dist

 

from typing import Dict, Any, Tuple, Union, Optional, List

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch, get_euler_xyz

import os, torch, gym, yaml, time

class Mk1ControlledTerrainTask(Mk1BaseClass):
    
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool, rl_device: str = "cuda:0") -> None:
        
        base_config = self._get_standard_config()
        
        config = join_configs(base_config, config)
        
        super().__init__(config, sim_device, graphics_device_id, headless, rl_device)
        
        # retreive pointers to simulation tensors
        self._get_gpu_gym_state_tensors()
        
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
            
        self.sim = sim
                    
        self.terrain_type = self.config["mk1_controlled_terrain"]["terrain"]["terrainType"] 
        if self.terrain_type=='plane':
            self._create_ground_plane()
        elif self.terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True 
        
        return sim
        
        
    def _create_trimesh(self):
        
        terrain_dict = self.config["mk1_controlled_terrain"]["terrain"]  
        
        self.terrain = Terrain(terrain_dict, num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = terrain_dict["staticFriction"]
        tm_params.dynamic_friction = terrain_dict["dynamicFriction"]
        tm_params.restitution = terrain_dict["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    
    
        
        
    def _create_envs(self, spacing: float, num_per_row: int) -> None:  
        """Create all the environments and initialize the agens in those environments

        Args:
            spacing (float): _description_
            num_per_row (int): _description_
        """
        
        
        
        if self.terrain_type == 'trimesh':
            terrain_dict = self.config["mk1_controlled_terrain"]["terrain"]  
        
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
            self.terrain_levels = torch.randint(0, terrain_dict["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.randint(0, terrain_dict["numTerrains"], (self.num_envs,), device=self.device)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.01
        
        
        # define plane on which environments are initialized
        env_lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)


        self.mk1_robot_asset = self.create_mk1_asset(self.config['mk1']['pure_shapes'])

        
        self.num_dof = self.gym.get_asset_dof_count(self.mk1_robot_asset) 
                
                
        self.robot_handles = []
        self.envs = [] 
        
        
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0,0.0, self.spawn_height)
        start_pose.r = gymapi.Quat(0, 0 , 1, 1)
        
        self._motor_efforts = self._create_default_effort_tensor(self.mk1_robot_asset)
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            
            
            if self.terrain_type == 'trimesh':  
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
                pos[2] += self.spawn_height
                start_pose.p = gymapi.Vec3(*pos)
            
            
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
            
            
            self._add_to_env(env_ptr, i , robot_handle)
            
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            
            
        # get all dofs and assign the action index to the dof name in the dof_name_index_dict
        self.dof_name_index_dict = self.gym.get_actor_dof_dict(env_ptr, robot_handle)
        
        # get all the rigid
        self.actor_rigid_body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, robot_handle)
        
        
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
        
        
        self.upper_body_joint_names = ['left_shoulder_a', 
                                       'right_shoulder_a',
                                       'left_shoulder_b',
                                       'right_shoulder_b',
                                       'right_arm_rotate',
                                       'left_arm_rotate',
                                       'right_elbow',
                                       'left_elbow'] # all the names of the upper body joints, that for instance should be used minimally when walking
        self.upper_body_joint_indices = torch.LongTensor([ self.dof_name_index_dict[name] for name  in  self.upper_body_joint_names ])
         
        
        
        
    def reset_envs(self, env_ids: torch.Tensor) -> None:
        super().reset_envs(env_ids)  
        
        n_envs_reset =  len(env_ids)
        
        self.target_velocity[env_ids] = sample_tensor_dist(self.target_velocity_dist, sample_shape=(n_envs_reset, ), device = self.device)
        
        self.target_direction[env_ids, 0] = torch.normal(mean = self.x_direction_mean[env_ids], std = self.x_direction_std[env_ids])
        self.target_direction[env_ids, 1] = torch.normal(mean = self.y_direction_mean[env_ids], std = self.y_direction_std[env_ids])
    
        
        
    def _extract_params_from_config(self) -> None:
        """
        Extract local variables used in the sim from the config dict
        """
         
        assert self.config["mk1_controlled_terrain"] is not None, "The mk1_controlled config must be set on the task config file"
        
        reward_weight_dict = self.config["mk1_controlled_terrain"]["reward_weighting"]  
        
        self.energy_cost = reward_weight_dict["energy_cost"]
        self.death_cost = reward_weight_dict["death_cost"]
        self.alive_reward = float(reward_weight_dict["alive_reward"])
        self.upright_punishment_factor = reward_weight_dict["upright_punishment_factor"]
        self.jitter_cost = reward_weight_dict["jitter_cost"] /  self.action_size
        self.death_height = reward_weight_dict["death_height"]
        self.overextend_cost = reward_weight_dict["overextend_cost"]
        self.die_on_contact = reward_weight_dict.get("die_on_contact", True)
        self.die_not_upward = reward_weight_dict.get("die_not_upward", True)
        self.contact_punishment_factor = reward_weight_dict["contact_punishment"]
        self.slowdown_punish_difference = reward_weight_dict["slowdown_punish_difference"]
        
        self.target_velocity_factor = reward_weight_dict["target_velocity_factor"]
        self.target_facing_vel_factor = reward_weight_dict["target_facing_vel_factor"]
        
        self.forward_facing_vel_factor = reward_weight_dict["forward_facing_vel_factor"]
        
        self.arm_use_cost = reward_weight_dict['arm_use_cost']

        controls_dict = self.config['mk1_controlled_terrain']['controls']

        self.target_velocity_dist =  controls_dict['velocity']
        self.target_x_dist = controls_dict['direction_x']
        self.target_y_dist = controls_dict['direction_y']
        
        
        
    def allocate_buffers(self):
        """Allocate all important tensors and buffers
        """
        super().allocate_buffers()
        
        self.x_direction_mean = torch.ones((self.num_envs, ), device=self.device) * self.target_x_dist['mean']
        self.x_direction_std = torch.ones((self.num_envs, ), device=self.device) * self.target_x_dist['std']
        x_direction = torch.normal(mean = self.x_direction_mean, std = self.x_direction_std).unsqueeze(dim=1)
        
        self.y_direction_mean = torch.ones((self.num_envs, ), device=self.device) * self.target_y_dist['mean']
        self.y_direction_std = torch.ones((self.num_envs, ), device=self.device) * self.target_y_dist['std']
        y_direction = torch.normal(mean = self.y_direction_mean, std = self.y_direction_std).unsqueeze(dim=1)
        
        
        
        self.target_direction =  torch.cat((x_direction, y_direction), dim= 1)
        
        self.target_velocity = sample_tensor_dist(self.target_velocity_dist, sample_shape=(self.num_envs, ), device= self.device)
        
        
    def refresh_tensors(self):
        super().refresh_tensors()
        
        #self.gym.render_all_camera_sensors(self.sim)


    def _compute_robot_rewards(self) -> Tuple[torch.Tensor, torch.Tensor,]:
        """Compute the rewards and the is terminals of the step
        -> all the important variables are sampled using the self property
        
        Returns:
               Tuple[torch.Tensor, torch.Tensor]: 
                   reward: torch.Tensor shape(num_envs, )
                   has_fallen: torch.Tensor shape(num_envs, )
                   constituents: Dict[str, int] contains all the average values, that make up the reward (i.e energy_punishment, directional_reward)
        """

        torso_index = self.actor_rigid_body_dict['upper_torso']
        
        torso_rigid_body_state = self.rigid_body_state_tensor[ : ,torso_index]        
    
        # -------------- base reward for being alive --------------  
        
        reward = torch.ones_like(self.root_states[:, 0]) * self.alive_reward  
        
        
        quat_rotation = torso_rigid_body_state[: , 3:7]
         
        #  -------------- reward for an upright torso -------------- 
        
        # The upright value ranges from 0 to 1, where 0 is completely vertical and 1 is completely horizontal
        # Calulation explanation: 
        # take the first and the last value of the quaternion and take the quclidean distance
        # upright_value = torch.sqrt(torch.sum( torch.square(quat_rotation[:, 0:4:3]), dim= 1 ))
        upright_factor = torch.sqrt(torch.sum( torch.square(quat_rotation[:, 0:2]), dim= 1 ))
        
        upright_punishment = upright_factor* self.upright_punishment_factor * -1
        
        reward += upright_punishment
        
        #  -------------- Precalcs for forward heading reward and match target velocity reward------------- 
        
        
        euler_rotation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = get_euler_xyz(quat_rotation)
        
        linear_velocity_x_y = self.root_states[: , 7:9]
        pose_direction_in_deg_to_x = euler_rotation[0]
        
        # compute the normalized pose_direction_vector
        # compute the normalized vel_direction_vector
        
        
        # x_y_pose_direction_vector
        # unit vecor of heading when seen from above 
        # this unit vector makes little sense, when the robot is highly non vertical
        x_y_pose_direction = torch.concat((torch.cos(pose_direction_in_deg_to_x).unsqueeze(dim = 1), torch.sin(pose_direction_in_deg_to_x).unsqueeze(dim = 1)), dim = 1)
 
        
        
        x_y_vel_direction_normalized = batch_normalize_vector(linear_velocity_x_y)
        
        # The angle (in radians) between the pose direction and the velocity direction
        # In a perfect world clamping would not be necessary, but because of rounding errors it is
        angle_between_pose_and_vel = torch.acos(torch.clamp(batch_dot_product(x_y_vel_direction_normalized, x_y_pose_direction), min = -1,  max = 1))
        
        
        # ---------- reward for the heading in the forward direction ----------
        
        # if the angle is 0, the full target direction factor is given
        # if the angle is 180 deg -> actor is running backwards instad of forward, the di
        forward_facing_vel_reward = ((1.5707 -  angle_between_pose_and_vel) / 1.5707) * self.forward_facing_vel_factor
        
        forward_facing_vel_reward = torch.where(self.target_velocity == 0, torch.zeros_like(forward_facing_vel_reward), forward_facing_vel_reward)
        
        reward += forward_facing_vel_reward
    
    
        # ---------- reward for matching the target velocity ----------

        # compare the two_d_heading_direction with the linear_velocity_x_y using the angle between them
        # magnitude of the velocity (2 norm)
        vel_norm = get_batch_tensor_2_norm(linear_velocity_x_y)
        
        # positive is to fast and neg is to slow 
        vel_difference = vel_norm - self.target_velocity 
        
        vel_reward_factor = torch.where(vel_difference > 0 , compute_velocity_reward_factor(vel_difference, self.slowdown_punish_difference, self.target_velocity_factor), compute_velocity_reward_factor(- vel_difference, self.target_velocity, self.target_velocity_factor))
        
        # Only apply the matching target velocity reward, when the actor is upright and the velocity is in the right direction (here 0.5 read = 29 deg)
        target_velocity_reward = torch.where(torch.logical_and((1- upright_factor) > 0.7, angle_between_pose_and_vel < 0.5), vel_reward_factor, torch.zeros_like(reward))
    
        reward += target_velocity_reward  
        
        # ---------- reward for matching the target direction ----------
        
        normed_direction = batch_normalize_vector(self.target_direction)
        
        angle_between_vel_and_target = torch.acos(torch.clamp(batch_dot_product(x_y_vel_direction_normalized, normed_direction), min = -1, max = 1))
        
        target_facing_vel_reward = ((1.5707 -  angle_between_vel_and_target) / 1.5707) * self.forward_facing_vel_factor
        
        
        target_facing_vel_reward = torch.where(self.target_velocity == 0, torch.zeros_like(target_facing_vel_reward), target_facing_vel_reward)
        
        reward += target_facing_vel_reward
        
        
        
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
        
        # ------------- cost of usign arms ------------
         
        arm_use_punishment = torch.sum(self.actions[:, self.upper_body_joint_indices]) / self.upper_body_joint_indices.shape[0] * self.arm_use_cost
        
        reward -= arm_use_punishment
        
        # ------- termination due to bad posture 
        is_terminal_step =  torch.zeros_like(reward, dtype=torch.int8)
        
        if self.die_not_upward:
            is_terminal_step += torch.where(upright_factor > 0.3, torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        
        
        # ---------- has fallen or die on contact -------------
                
        terminations_height = self.death_height
        
        has_fallen = torch.zeros_like(reward, dtype=torch.int8)
        has_fallen = torch.where(self.root_states[:, 2] < terminations_height, torch.ones_like(reward,  dtype=torch.int8) , torch.zeros_like(reward, dtype=torch.int8))
        
        is_terminal_step += has_fallen
        
        summed_contact_forces = torch.sum(self.contact_forces, dim= 2) # sums x y and z components of contact forces together
        
        summed_contact_forces[:,self.left_foot_index] = 0.0
        summed_contact_forces[:, self.right_foot_index] = 0.0
        
        total_summed_contact_forces = torch.sum(summed_contact_forces, dim=1) # sum all the contact forces of the other indices together, to see if there is any other contact other than the feet
        
        has_contact = torch.where(total_summed_contact_forces > torch.zeros_like(total_summed_contact_forces), torch.ones_like(reward, dtype=torch.int8), torch.zeros_like(reward, dtype=torch.int8))
        
        if self.die_on_contact:
            is_terminal_step += has_contact
        else:
            n_times_contact = (summed_contact_forces > 0 ).to(dtype=torch.float32).sum(dim=1)
            
            contact_punishment = n_times_contact * self.contact_punishment_factor
            
            reward -= contact_punishment
        
        # ------------- cost for dying ----------
        # root_states[:, 2] defines the y positon of the root body 
        reward = torch.where(is_terminal_step == 1, - 1 * torch.ones_like(reward) * self.death_cost, reward)
        
    
        
        # average rewards per step
         
        upright_punishment = float(torch.mean(upright_punishment).item())
        target_velocity_reward = float(torch.mean(target_velocity_reward).item())
        jitter_punishment = - float(torch.mean(jitter_punishment).item())
        energy_punishment = - float(torch.mean(energy_punishment).item())
        arm_use_punishment = - float(torch.mean(arm_use_punishment).item())
        forward_facing_vel_reward = float(torch.mean(forward_facing_vel_reward).item())
        overextend_punishment = - float(torch.mean(overextend_punishment).item())
        if not self.die_on_contact:
            contact_punishment = -float(torch.mean(contact_punishment).item())
        else:
            contact_punishment = 0.0
        
        total_avg_reward = float(torch.mean(reward).item())
        
        reward_constituents = {
                                'alive_reward': self.alive_reward,
                                'upright_punishment':  upright_punishment, 
                                'jitter_punishment':   jitter_punishment,
                                'forward_facing_vel_reward': forward_facing_vel_reward, 
                                'target_velocity_reward': target_velocity_reward,
                                'energy_punishment':   energy_punishment,
                                'arm_use_punishment': arm_use_punishment,
                                'overextend_punishment': overextend_punishment,
                                'contact_punishment': contact_punishment,
                                'total_reward': total_avg_reward
                            }
        
        
        return (reward, is_terminal_step, reward_constituents)
            
    
    
    def _add_to_env(self, env_ptr, env_id: int, robot_handle): 
        """During the _create_envs this is called to give mk1_envs the ability to add additional things to the environment

        Args:
            env_ptr (_type_): pointer to the env
        """
        
        # camera_props = gymapi.CameraProperties()
        # camera_props.width = 128
        # camera_props.height = 128
        # camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        # 
        # local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(0,0.4,1.6)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
        #  
        # self.gym.attach_camera_to_body(camera_handle, env_ptr,  robot_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        
        pass
    
    def _get_standard_config(self) -> Dict:
        """Get the dict of the standard configuration

        Returns:
            Dict: Standard configuration
        """
        dirname = os.path.dirname(__file__)
        base_config_path = os.path.join(dirname, 'config_mk1_terrain.yaml')
        
          # open the config file 
        with open(base_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Base Config : {base_config_path} not found")
            
            
    def get_obs(self) -> Tuple[ Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return all the observations made by the actor and the critic

        Returns:
            Tuple[str, torch.Tensor]: _description_
        """
        
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
        
        self.actor_obs["command"][:, 1:3] = self.target_direction
        self.actor_obs["command"][:, 0] = self.target_velocity
        
    
    
    def get_num_playable_actors_per_env(self) -> int:
        """Return the amount of actors each environment has, this only includes actors, that are playable
        This distincion is stupid and only exists, because isaacgym currently does anot support any way of adding objects to environments, that are not actors

        Returns:
            int
        """
        return self.get_num_actors_per_env()
    
    
    def get_num_actors_per_env(self) -> int:
        """Get the total amount of actor per environment this includes non active actors like boxes or other inaminate matter

        Returns:
            int
        """
        
        return 1
    
    def _get_actor_observation_spaces(self) -> MultiSpace:
        """Define the different observation the actor of the agent
         (this includes linear observations, viusal observations, commands)
         
         The observations will later be combined with other inputs like commands to create the actor input space
        
        This is an asymmetric actor critic implementation  -> The actor observations differ from the critic observations
        and unlike the critic inputs the actor inputs have to be things that a real life robot could also observe in inference

        Returns:
            MultiSpace: [description]
        """
        num_actor_obs = 142
        return  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, )),
            "command": gym.spaces.Box(low= -3.0, high= 5.0, shape= (3, ))
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
        
         
def compute_velocity_reward_factor(abs: torch.Tensor, zero_x_cord: Union[float, torch.Tensor], factor: Union[float, torch.Tensor]) -> torch.Tensor:
    """compute a function for the velocity reward factor
    https://www.geogebra.org/graphing/f2qxcw85

    Args:
        abs (torch.Tensor): the absolute tensor (x is only positive)
        zero_x_cord (torch.Tensor): the coordinate where the reward is zero
        factor (Union[float, torch.Tensor]): the y multiplication. This values is achieved for abs == 0
        
        f(x)=(((1)/(x ((1)/(2 zerop))+0.5))-1) * factor
    Returns:
        torch.Tensor: y
    """
    return  (1.0 / (abs * (1.0 / (2.0 * zero_x_cord))+ 0.5) - 1.0) * factor
     
    
 
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
     
    
    linear_actor_obs = torch.cat((sensor_states.view(root_states.shape[0], -1), dof_pos, dof_vel, dof_limits_upper.tile(( root_states.shape[0], 1)),dof_limits_lower.tile(( root_states.shape[0],1 )),  dof_force, ang_velocity, torso_rotation, actions, torso_position), dim=-1)
    
    linear_critic_obs = torch.cat(( velocity, torso_position), dim=-1)
    
    return  linear_actor_obs,   linear_critic_obs



