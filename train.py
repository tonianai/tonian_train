from typing import Dict, Optional

from tonian.tasks.cartpole.cartpole_task import Cartpole 

import yaml, argparse, os

from tonian.training.algorithms import PPOAlgorithm
from tonian.training.policies import  build_A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger, TensorboardLogger
from tonian.common.config_utils import create_new_run_directory, task_from_config
from tonian.common.utils import set_random_seed, join_configs


def train(config_path: str, 
          seed: int = 0,  
          config_overrides: Dict = {}, 
          headless: bool = False, 
          batch_id: Optional[str] = None,
          model_out_name: Optional[str] = None, 
          verbose: bool = True,
          max_steps: Optional[int] = None):
    """Train the given config

    Args:
        config_path (str): The path to the base config fsile, describing task, policy and algo
        seed (int): the deterministic seed
        config_overrides (Dict): values that should be overwritten of the config
        determines whether the robots should be shown
        batch_id (Optional[str], optional): Name of the batch, in order to cluster runs together. Defaults to None.

    Raises:
        FileNotFoundError: the config was not found
    """
    set_random_seed(int(seed))
        
            
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
        
    config = join_configs(config, config_overrides)
    
    
    task = task_from_config(config['task'], headless)

    policy = build_A2CSequentialLogStdPolicy(config['policy'], 
                                            actor_obs_space=task.actor_observation_spaces, 
                                            critic_obs_space=task.critic_observation_spaces, 
                                            action_space= task.action_space)
    
    policy.to('cuda:0')
    
    
    if 'start_model' in config:
    
        # load the model to the policy 
    
        policy.load(config['start_model'])
        
    

    # create the run folder here
    run_folder_name, run_id = create_new_run_directory(config, batch_id)
        
    logger = TensorboardLogger(run_folder_name, run_id)

    logger.log_config('policy',config['policy'])
    logger.log_config('algo',config['algo'])
    logger.log_config('task', config['task'])

    algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy, verbose, model_out_name)

    algo.train(max_steps)
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, default = 0, help="Seed for running the env")
    ap.add_argument("-cfg", required= False, default= 'cfg/mk1-terrain-test.yaml', help="path to the config")
    ap.add_argument("-batch_id", required= False, default= None,  help="name of the running batch")
    ap.add_argument("-model_out", required=False,default= None, help="The name under wich the model will be registered in the models folder" )
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')

    ap.set_defaults(feature= False)
    
    args = vars(ap.parse_args())
    
    train(args['cfg'], args.get('seed', 0), {}, args['headless'], args.get('batch_id', None), args.get('model_out', None), True)


