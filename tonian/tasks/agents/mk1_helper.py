 

from typing import Tuple, List
from isaacgym import gymapi
from tonian.common.spaces import MultiSpace

import torch, os, gym



    
    
# The parts of the robot, that should get a force sensor    
parts_with_force_sensor = ['upper_torso' , 'foot', 'foot_2', 'forearm', 'forearm_2' ]



def create_mk1_asset(gym, sim):
    """Create the Mk1 Robot with all sensors and setting in the simulation and return the asset

    Args:
        gym (_type_): isaacgym gym reference
        sim (_type_): isaacgym sim reference
    """
    
    
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf/mk-1/")
        
    mk1_robot_file = "robot.urdf"
        
    asset_options = gymapi.AssetOptions()
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    # asset_options.fix_base_link = False
    # asset_options.disable_gravity = False
    
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.fix_base_link = False
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False
    
    
    mk1_robot_asset = gym.load_asset(sim, asset_root, mk1_robot_file, asset_options)
    
    
    sensor_pose = gymapi.Transform()
    
    
    
    
    parts_idx = [gym.find_asset_rigid_body_index(mk1_robot_asset, part_name) for part_name in parts_with_force_sensor]
    
    print(parts_idx)
    
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




num_critic_obs = 126
critic_obs_space = MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_critic_obs, ))
        })
    
num_actor_obs = 128
actor_obs_space =  MultiSpace({
            "linear": gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actor_obs, ))
        })

num_actions = 17
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions, )) 