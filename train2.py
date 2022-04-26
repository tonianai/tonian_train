
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1.mk1_walking.mk1_walking_task import Mk1WalkingTask

import yaml

from tonian.training2.algorithms.algorithms import PPOAlgorithm
from tonian.training2.networks import A2CSequentialNetLogStd, build_A2CSequientialNetLogStd
from tonian.training2.policies import A2CSequentialLogStdPolicy
from tonian.common.logger import DummyLogger

from tonian.common.utils import set_random_seed


config_path = './cfg/mk1-walking-test2.yaml'
    

set_random_seed(0)
        
# open the config file 
with open(config_path, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:    
        raise FileNotFoundError( f"File {config_path} not found")

task = Mk1WalkingTask(config['task'], 'cuda:0', 0, False)

network = build_A2CSequientialNetLogStd(config['policy'], 
                                        actor_obs_space=task.actor_observation_spaces, 
                                        critic_obs_space=task.critic_observation_spaces, 
                                        action_space= task.action_space)

policy = A2CSequentialLogStdPolicy(network)


logger = DummyLogger()

logger.log_config('policy',config['policy'])
logger.log_config('algo',config['algo'])

algo = PPOAlgorithm(task, config['algo'], 'cuda:0', logger, policy)

algo.train()



