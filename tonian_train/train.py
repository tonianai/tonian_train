from typing import Dict, Optional
 

from testing_env.cartpole.cartpole_task import Cartpole 

import yaml, argparse, os, csv

from tonian_train.algorithms.sequential_ppo import SequentialPPO
from tonian_train.policies import  build_a2c_sequential_policy
from tonian_train.common.logger import DummyLogger, TensorboardLogger, CsvFileLogger, LoggerList, CsvMaxFileLogger, WandbLogger
from tonian_train.common.spaces import MultiSpace
 
from tonian_train.common.config_utils import create_new_run_directory
from tonian_train.common.utils import set_random_seed, join_configs
from tonian_train.tasks import TaskBuilder


def train(config_path: str, 
          task_factory: TaskBuilder,
          seed: int = 0,  
          config_overrides: Dict = {}, 
          headless: bool = False, 
          batch_id: Optional[str] = None,
          model_out_name: Optional[str] = None, 
          model_out_if_better: Optional[bool] = None,
          verbose: bool = True,
          max_steps: Optional[int] = None,
          project_name: str = "training_alg",
          save_obs: bool = False):
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
    
    
    task = task_factory(config['task'], headless, seed = seed)
     
     
    policy = build_a2c_sequential_policy(config['policy'], 
                                            obs_space=task.observation_space,  
                                            action_space= task.action_space)
    
    policy.to('cuda:0')
    
    reward_to_beat = None
    
    if 'start_model' in config:
    
        # load the model to the policy 
    
        policy.load(config['start_model'])
        
        if config.get('use_last_lr', False):
            file = open(os.path.join(config['start_model'], 'last_logs.csv'))
            csvreader = csv.reader(file)
            
            header = next(csvreader)
            index_of_lr = header.index('info/last_lr')
            
            last_lr = next(csvreader)[index_of_lr]
            
            config['algo']['learning_rate'] = last_lr
            
    if model_out_name and config.get('only_save_when_beaten', False):
        
        file_path = os.path.join('models', model_out_name, 'max_logs.csv')
        
        if os.path.exists(file_path):
        
            file = open(file_path)
            csvreader = csv.reader(file)
            
            header = next(csvreader)
            index_of_reward = header.index('run/episode_rewards')
            
            biggest_reward = next(csvreader)[index_of_reward]
            
            reward_to_beat = biggest_reward
        
        
    

    # create the run folder here
    run_folder_name, run_id = create_new_run_directory(config, batch_id)
         

    csv_logger = CsvFileLogger(run_folder_name, run_id)
    max_csv_logger = CsvMaxFileLogger(run_folder_name, run_id)
     
    if batch_id is None:
        wand_name = config['name']+ '_' +str(run_id)
    else: wand_name = batch_id + ' '+ str(run_id)
    
    
    task_config =  task.config
    full_config = {'algo': config['algo'], 'task': task_config}
    
    wandb_logger = WandbLogger(wand_name, project_name)
    wandb_logger.log_config('algo',full_config)

    
    logger_list = [ csv_logger, max_csv_logger, wandb_logger]
    
        
    
    if model_out_name:
        out_path = os.path.join('models', model_out_name)
        
        os.makedirs(out_path, exist_ok=True)
        logger_list.append(CsvFileLogger(out_path, run_id))
        logger_list.append(CsvMaxFileLogger(out_path, run_id))
        
    
    
    
    logger = LoggerList( logger_list, run_id, run_folder_name)

    algo = SequentialPPO(task, config['algo'], 'cuda:0', logger, policy, verbose, model_out_name, reward_to_beat)

    algo.train(max_steps, save_obs = save_obs)