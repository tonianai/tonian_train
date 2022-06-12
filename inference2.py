
from tonian.tasks.cartpole.cartpole_task import Cartpole

import yaml, os, argparse, torch
from tonian.common.config_utils import  task_from_config
from tonian.training.policies import  build_A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger
from tonian.training.algorithms import PPOAlgorithm
import yaml, argparse, os

from tonian.training.algorithms import PPOAlgorithm
from tonian.training.policies import  build_A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger, TensorboardLogger
from tonian.common.config_utils import create_new_run_directory, task_from_config
from tonian.common.utils import set_random_seed, join_configs


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-run_dir', required=True, help= 'path to the run directory')
    
    args = vars(ap.parse_args())
    
    run_dir = args['run_dir'] 
     
    if not os.path.exists(run_dir):
        print(f"Run folder {run_dir}")
        raise FileNotFoundError("The batch path does not exist")
    
    
    config_path = os.path.join(run_dir, 'config.yaml')
    
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {config_path} not found")
        
        
    task = task_from_config(config['task'])
     
    policy = build_A2CSequentialLogStdPolicy(config['policy'], 
                                            actor_obs_space=task.actor_observation_spaces, 
                                            critic_obs_space=task.critic_observation_spaces, 
                                            action_space= task.action_space)
    
    policy.to('cuda:0')
    
    # load the model to the policy 
    
    policy.load(os.path.join(run_dir,'saves', 'best_model'))
    
    

    # create the run folder here
    run_folder_name, run_id = create_new_run_directory(config)
        
    logger = TensorboardLogger(run_folder_name, run_id)

    logger.log_config('policy',config['policy'])
    logger.log_config('algo',config['algo'])
    logger.log_config('task', config['task'])
    
    
    
    for name, param in policy.named_parameters():
        print(name)
        print(param)
        
      

    algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy, True)
    
    #algo.init_run()
    
    #while True:
    #    algo.play_steps()
    algo.train()