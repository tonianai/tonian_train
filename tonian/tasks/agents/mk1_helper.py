 

from typing import Tuple, List
from isaacgym import gymapi

import torch, os


# The parts of the robot, that should get a force sensor    
parts_with_force_sensor = ['foot', 'foot_2', 'forearm', 'forearm_2']


def create_mk1_in_envs(gym, sim, env_ptrs: List) -> Tuple[List, List, List]:
    """Create the mk1 robots in the environment with all sensors attached and activated 

    Args:
        gym (_type_): _description_
        sim (_type_): _description_
        env_ptrs (List): _description_

    Returns:
        Tuple[List, List, List]: (robot_handles, lower_limits, upper_limits)
            - robot handles corresponds with index to env_ptrs  => len(robot_handles) == len(env_ptrs)
            - upper and lower limit arrays correspons to the dof => len(upper_limits) == num_dof
    """
    
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0.0,0.0, 1.80)
    start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    

    mk1_robot_asset = create_mk1_asset(gym, sim)

        
    num_dof = gym.get_asset_dof_count(mk1_robot_asset) 
    
    robot_handles = []
                

    for i, env_ptr in enumerate(env_ptrs):
        
        robot_handle = gym.create_actor(env_ptr, mk1_robot_asset, start_pose, "mk1", i, 1, 0)
        
        dof_prop = gym.get_actor_dof_properties(env_ptr, robot_handle)
        gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            
            # TODO: Maybe change dof properties dof_probs)
            
        
        robot_handles.append(robot_handle)
            
        
            
        # take the last one as an example (All should be the same)
        dof_prop = gym.get_actor_dof_properties(env_ptr, robot_handle)
        
        dof_limits_lower = []
        dof_limits_upper = []
        for j in range(num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                dof_limits_lower.append(dof_prop['upper'][j])
                dof_limits_upper.append(dof_prop['lower'][j])
            else:
                dof_limits_lower.append(dof_prop['lower'][j])
                dof_limits_upper.append(dof_prop['upper'][j])
    
    return robot_handle, num_dof ,dof_limits_lower, dof_limits_upper
    
    


def create_mk1_asset(gym, sim):
    """Create the Mk1 Robot with all sensors and setting in the simulation and return the asset

    Args:
        gym (_type_): isaacgym gym reference
        sim (_type_): isaacgym sim reference
    """
    
    
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf/mk-1/")
        
    mk1_robot_file = "robot.urdf"
        
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    
    
    mk1_robot_asset = gym.load_asset(sim, asset_root, mk1_robot_file, asset_options)
    
    
    sensor_pose = gymapi.Transform()
    
    
    
    
    parts_idx = [gym.find_asset_rigid_body_index(mk1_robot_asset, part_name) for part_name in parts_with_force_sensor]
    for part_idx in parts_idx:
        gym.create_asset_force_sensor(mk1_robot_asset, part_idx, sensor_pose)
        
    return mk1_robot_asset


@torch.jit.script
def compute_linear_robot_observations(root_states: torch.Tensor, 
                                sensor_states: torch.Tensor, 
                                dof_vel: torch.Tensor, 
                                dof_pos: torch.Tensor, 
                                dof_limits_lower: torch.Tensor,
                                dof_limits_upper: torch.Tensor,
                                dof_force: torch.Tensor,
                                actions: torch.Tensor,
                                initial_heading: torch.Tensor
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
    
    # print("Torso Position")
    # print(torso_position.shape)
    # print(torso_rotation.shape)
    # print(velocity.shape)
    # print(ang_velocity.shape)
    # print(sensor_states.shape)
    
    
    
    # todo add some other code to deal with initial information, that might be required
    
    
    # todo: the actor still needs a couple of accelerometers
    linear_actor_obs = torch.cat((sensor_states.view(root_states.shape[0], -1), dof_pos, dof_vel, dof_force, ang_velocity, torso_rotation, actions), dim=-1)
    
    
    # print('actor_obs')
    # print(linear_actor_obs.shape)
    linear_critic_obs = torch.cat((linear_actor_obs, torso_rotation, velocity, torso_position, actions), dim=-1)
    
    # print('critic_obs')
    # print(linear_critic_obs.shape)
    return  linear_actor_obs,   linear_critic_obs