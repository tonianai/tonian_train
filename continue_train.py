"""
Continue training on a previously stored run
    
"""
from tonian.algorithms.ppo import PPO
from tonian.tasks.cartpole.cartpole_task import Cartpole # import for 

import yaml, os, argparse
from tonian.common.logger import TensorboardLogger
from tonian.common.utils.config_utils import get_run_index, policy_from_config, task_from_config, algo_from_config
from tonian.common.utils.utils import set_random_seed

import torch

if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=True, help="Name of the environment you want to run : optional run number eg. Cartpole:10")
    ap.add_argument("-n_steps", required=False, help="The amount of steps the training should continue", default=1e9)
    ap.add_argument("-device", required=False, help="The device used for training etc.", default="cuda:0")
    
    
    
    args = vars(ap.parse_args())
    device = args['device']
    n_rollout_steps = args['n_steps']
    env_name = ''
    run_nr = ''
    
    run_base_folder= 'runs/'
    
    if ':' in args['run']:
        # the run nr is set in the run string 
        args_arr = args['run'].split(':')
        env_name = args_arr[0]
        run_nr = args_arr[1]
    else:
        # the run number is not set in the run string
        # -> use the most recent one
        env_name = args['run']
        run_nr = get_run_index(run_base_folder + env_name) - 1
        
    run_folder = run_base_folder + env_name + '/'+ str(run_nr)
    
    if not os.path.exists(run_folder):
        raise FileNotFoundError("The run path does not exist")
    
    device = "cuda:0"
    
    config_path = run_folder + '/config.yaml'
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {config_path} not found")
    
    if "seed" in config:   
        set_random_seed(config["seed"])
    
    
    task = task_from_config(config['task'])
    task.is_symmetric = False
    
    policy = policy_from_config(config['policy'], task)
    
    # get the highest step stored policy
    
    saves_folder = os.path.join(run_folder, 'saves')
    saves_in_directory = os.listdir(saves_folder)
    
    
    if len(saves_in_directory) == 0:
        print("WARNING: The run has no saves policy ---- starting from 0!!!")
    elif len(saves_in_directory) == 1:
        # only one policy -> use that 
        policy.load(os.path.join(saves_folder, saves_in_directory[0]))
    else:
        # multiple policies -> choose the last or the closest to the goal one    
        step_nr = max([int(file_name.split('.')[0]) for file_name in saves_in_directory])
        file_name = str(step_nr) + '.pth'
        policy.load(os.path.join(saves_folder, file_name))
        
    
    logger = TensorboardLogger(run_folder)
        
        
    algo = algo_from_config(config["algo"], task, policy, device, logger)

    algo.learn(total_timesteps=1e10)
    task.close()
    
    