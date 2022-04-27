from typing import Dict, Optional

from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1.mk1_walking.mk1_walking_task import Mk1WalkingTask

import yaml, argparse

from tonian.training.algorithms import PPOAlgorithm
from tonian.training.networks import A2CSequentialNetLogStd, build_A2CSequientialNetLogStd
from tonian.training.policies import A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger, TensorboardLogger
from tonian.common.config_utils import create_new_run_directory
from tonian.common.utils import set_random_seed, join_configs


def train(config_path: str, seed: int = 0,  config_overrides: Dict = {}, headless: bool = False, batch_id: Optional[str] = None, verbose: bool = True ):
    """Train the given config

    Args:
        config_path (str): The path to the base config file, describing task, policy and algo
        seed (int): the deterministic seed
        config_overrides (Dict): values that should be overwritten of the config
        determines whether the robots should be shown
        batch_id (Optional[str], optional): Name of the batch, in order to cluster runs together. Defaults to None.

    Raises:
        FileNotFoundError: the config was not found
    """
    set_random_seed(seed)
        
            
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
        
    config = join_configs(config, config_overrides)
    
        
    task = Mk1WalkingTask(config['task'], 'cuda:0', 0, headless)

    network = build_A2CSequientialNetLogStd(config['policy'], 
                                            actor_obs_space=task.actor_observation_spaces, 
                                            critic_obs_space=task.critic_observation_spaces, 
                                            action_space= task.action_space)

    policy = A2CSequentialLogStdPolicy(network)

    # create the run folder here
    run_folder_name, run_id = create_new_run_directory(config, batch_id)
        
    logger = TensorboardLogger(run_folder_name, run_id)

    logger.log_config('policy',config['policy'])
    logger.log_config('algo',config['algo'])
    logger.log_config('task', config['task'])

    algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy, verbose)

    algo.train()
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, default = 0, help="Seed for running the env")
    ap.add_argument("-cfg", required= True, help="path to the config")
    ap.add_argument("-batch_id", required= False, default= None,  help="name of the running batch")
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')
    ap.set_defaults(feature= False)
    
    args = vars(ap.parse_args())
    
    train(args['cfg'], args.get('seed', 0), {}, args['headless'], args.get('batch_id'), None)



