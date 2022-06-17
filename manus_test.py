from typing import Dict, Optional

from tonian.tasks.cartpole.cartpole_task import Cartpole 

import yaml, argparse, os, csv

from tonian.training.algorithms import PPOAlgorithm
from tonian.training.policies import  build_A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger, TensorboardLogger, CsvFileLogger, LoggerList, CsvMaxFileLogger
from tonian.common.config_utils import create_new_run_directory, task_from_config
from tonian.common.utils import set_random_seed, join_configs


def train_ahole(config_path: str, 
          seed: int = 0,  
          config_overrides: Dict = {}, 
          headless: bool = False, 
          batch_id: Optional[str] = None,
          model_out_name: Optional[str] = None, 
          model_out_if_better: Optional[bool] = None,
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
            
    if model_out_name and config.get('use_last_lr', False):
        
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
    
    print("run_id")
    print(run_id)
        
    tb_logger = TensorboardLogger(run_folder_name, run_id)

    tb_logger.log_config('policy',config['policy'])
    tb_logger.log_config('algo',config['algo'])
    tb_logger.log_config('task', config['task'])


    csv_logger = CsvFileLogger(run_folder_name, run_id)
    max_csv_logger = CsvMaxFileLogger(run_folder_name, run_id)
    
    
    logger_list = [tb_logger, csv_logger, max_csv_logger]
    
    
    
    if model_out_name:
        out_path = os.path.join('models', model_out_name)
        
        os.makedirs(out_path, exist_ok=True)
        logger_list.append(CsvFileLogger(out_path, run_id))
        logger_list.append(CsvMaxFileLogger(out_path, run_id))
        
    
    
    logger = LoggerList( logger_list, run_id, run_folder_name)

    algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy, verbose, model_out_name, reward_to_beat)

    algo.train(100000000)
    
    print("NEXT ONE")
    
    
    run_folder_name, run_id = create_new_run_directory(config, batch_id)
    
    
    print("run_id")
    print(run_id)
        
    tb_logger = TensorboardLogger(run_folder_name, run_id)

    tb_logger.log_config('policy',config['policy'])
    tb_logger.log_config('algo',config['algo'])
    tb_logger.log_config('task', config['task'])


    csv_logger = CsvFileLogger(run_folder_name, run_id)
    max_csv_logger = CsvMaxFileLogger(run_folder_name, run_id)
    
    
    logger_list = [tb_logger, csv_logger, max_csv_logger]
    
    logger = LoggerList( logger_list, run_id, run_folder_name)
    
    
    
    if model_out_name:
        out_path = os.path.join('models', model_out_name)
        
        os.makedirs(out_path, exist_ok=True)
        logger_list.append(CsvFileLogger(out_path, run_id))
        logger_list.append(CsvMaxFileLogger(out_path, run_id))
    
    algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy, verbose, model_out_name, reward_to_beat)

    
    
    algo.train(100000000)
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, default = 0, help="Seed for running the env")
    ap.add_argument("-cfg", required= False, default= 'cfg/mk1-running-test.yaml', help="path to the config")
    ap.add_argument("-batch_id", required= False, default= None,  help="name of the running batch")
    ap.add_argument("-model_out", required=False,default= None, help="The name under wich the model will be registered in the models folder" )
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')

    ap.set_defaults(feature= False)
    
    args = vars(ap.parse_args())
    
    train_ahole(args['cfg'], args.get('seed', 0), {}, args['headless'], args.get('batch_id', None), args.get('model_out', None),  True)

