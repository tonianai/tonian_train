
from tonian.tasks.cartpole.cartpole_task import Cartpole

from typing import Dict, List, Optional

import yaml

from tonian.common.utils import set_random_seed, join_configs
from tonian.common.config_utils import task_from_config, algo_from_config, policy_from_config, create_new_run_directory
from tonian.common.logger import TensorboardLogger

import yaml, argparse


def train(args: Dict, verbose: bool = True,  early_stopping: bool = False, early_stop_patience = 1e8, config_overrides: Dict = {}, batch_id: Optional[str] = None):
    """Train an environment given a config

    Args:
        args (Dict): arguments given via console
            required args['cfg'] 

    """
    
    args_seed = None
    
    if args['seed'] is not None:
        args_seed = int(args['seed'])
    
     
    
    headless =  'headless' in args and args['headless']
     
    device = "cuda:0"
    config_path = args['cfg']
      
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
      
    config = join_configs(config, config_overrides)
    
    if args_seed is not None:
        set_random_seed(args_seed)
        config["algo"]["seed"] = args_seed
    elif "seed" in config:   
        set_random_seed(config["seed"])
    
    # create the run folder here
    run_folder_name, run_id = create_new_run_directory(config, batch_id)
     
    logger = TensorboardLogger(run_folder_name, run_id)
    
    logger.log_config('policy',config['policy'])
    logger.log_config('algo',config['algo'])
    
    task = task_from_config(config["task"], headless= headless)
    
    # log after the task was created, since the task joins all the default config files with this file
    logger.log_config_items('task',task.config)
    
    policy = policy_from_config(config["policy"], task) 
    
    print("Model's state_dict:")
    for param_tensor in policy.state_dict():
        print(param_tensor, "\t", policy.state_dict()[param_tensor].size())
        
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in policy.optimizer.state_dict():
        print(var_name, "\t", policy.optimizer.state_dict()[var_name])
    
    if 'start_model' in config:
        policy.load(config['start_model'])
        
        
    
    algo = algo_from_config(config["algo"], task, policy, device, logger)
    
    
    algo.learn(total_timesteps=1e10, verbose = verbose, early_stopping = early_stopping, early_stopping_patience= early_stop_patience)
    
    task.close()
    

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, help="Seed for running the env")
    ap.add_argument("-cfg", required= True, help="path to the config")
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')
    ap.set_defaults(feature= False)
    
    args = vars(ap.parse_args())
    train(args, early_stopping= True, early_stop_patience=5e7)
    
   
    
    
    
        
