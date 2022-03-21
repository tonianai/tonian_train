

from typing import Dict

from tonian.common.logger import BaseLogger
from tonian.tasks.walking.walking_task import WalkingTask
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1_walking.mk1_walking_task import Mk1WalkingTask
from tonian.algorithms.base_algorithm import BaseAlgorithm
from tonian.algorithms.ppo import PPO
from tonian.policies.policies import SimpleActorCriticPolicy, ActorCriticPolicy

from gym.spaces import space
from tonian.tasks.base.command import Command
from tonian.tasks.base.vec_task import MultiSpace, VecTask
from tonian.common.schedule import Schedule

import torch
import torch.nn as nn
import os, yaml

def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        raise FileNotFoundError()
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base

                    
def parseActvationFunction(string: str):
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    return activations[string]


def task_from_config(config: Dict, headless: bool = False) -> VecTask:
    
    name_to_task_map=  {
     "00_cartpole": Cartpole,
     "01_walking": WalkingTask,
     "02_mk1_walking": Mk1WalkingTask
    }
    return name_to_task_map[config["name"]](config, sim_device="gpu", graphics_device_id=0, headless= headless)
    
def policy_from_config(config: Dict, env: VecTask) -> ActorCriticPolicy:
    
    actor_obs_spaces = env.actor_observation_spaces
    critic_obs_spaces = env.critic_observation_spaces
    action_space = env.action_space
    
    assert "lr" in config.keys(), "The learning rate must be specified in the config file"
    
    
    lr = Schedule(config["lr"])
    if config['name'] == "SimpleActorCritic":
        
        activation_fn = nn.Tanh
        if "activation_fn" in config:
            activation_fn = parseActvationFunction(config["activation_fn"])
        
        if "actor_hidden_layers" in config:    
            actor_hidden_layers = tuple(config['actor_hidden_layers'])
        else:
            actor_hidden_layers = (128, 128, 128)
        
        if "critic_hidden_layers" in config:    
            critic_hidden_layers = tuple(config['critic_hidden_layers'])
        else:
            critic_hidden_layers = (128,128,128)
            
        if "device" in config:
            device = config["device"]
        else:
            device = "cuda:0"
             
            
        # create a simple actor critic from the params
        return SimpleActorCriticPolicy(actor_obs_space= actor_obs_spaces,
                                       critic_obs_space= critic_obs_spaces,
                                       lr_schedule= lr,
                                       action_space=action_space,
                                       activation_fn_class=activation_fn,
                                       actor_hidden_layer_sizes=actor_hidden_layers,
                                       critic_hiddent_layer_sizes=critic_hidden_layers,
                                       device= device
                                       )
        # TODO: Make more stuff configurable
    else:
        raise Exception("Not supported Policy name from Configuration File")    
    
    pass

def algo_from_config(config: Dict, env: VecTask, policy: ActorCriticPolicy, device: str, logger: BaseLogger) -> BaseAlgorithm:
    
    if config['name'] == 'PPO':
        return PPO(env, config, policy, device, logger= logger)
    else:
        raise Exception("Not supported Alogorithm name from Configuration File")
    
    pass



def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        os.makedirs(base_folder_name)
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base
        
    

def create_new_run_directory(config: Dict) -> str:
    """Create a new run directory and store the given config in the directory
    Args:
        config (Dict): The config file, that contains all the important info to recreate, or continue this run
         
 
    Returns:
        str: run_folder_name
    """
    
    task_name = config['task']['name']
    
    # the name where  the network and log files will be stored about this run
    run_base_folder= f'runs/{task_name}'
       
    run_index = get_run_index(run_base_folder)
        
    run_folder_name = run_base_folder + "/"+ str(run_index)
     
    # create the run folder
    os.makedirs(run_folder_name)
    # create the run saves folder
    os.makedirs(run_folder_name + "/saves")
    # create the run logs folder
    os.makedirs(run_folder_name + "/logs")
    # save the config in the run folder
    
    with  open(f"{run_folder_name}/config.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=True)
        
    return run_folder_name
    
    
def join_configs(base_config: Dict, config: Dict) -> Dict:
    """Joins two configuration files into one

    Args:
        base_config (Dict): The base config
        config (Dict): This config can override values of the base config

    Returns:
        Dict: [description]
    """
    # idea go through all the values and join or override if the value is not a dict
    # if the value is a dict recursevely call this function
    
    final_dict = base_config.copy()
    
    for key, value in config.items():
        
        if isinstance(value, Dict):
            
            # check if it is in the base
            if key in final_dict.keys():
                # join the dicts using a recursive call
                final_dict[key] = join_configs(final_dict[key], value)
            else:
                final_dict[key] = value
            
        else:
            final_dict[key] = value
        
    
    return final_dict


             
            