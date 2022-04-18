

from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1.mk1_walking.mk1_walking_task import Mk1WalkingTask

from tonian.common.config_utils import task_from_config

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner 
 
import yaml


from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext

class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name):
        self.env = env_configurations.configurations[config_name]['env_creator']()

    def step(self, action):
        
        obs, rewards, dones, infos, _ =   self.env.step(action)
        
        return obs[0]['linear'], rewards, dones, infos

    def reset(self):
        init_obs =  self.env.reset()[0]['linear']
        return init_obs
        
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space['linear']

        return info






def create_env_func():

    config_path = './cfg/mk1-walking-test.yaml'
      
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")

    task = Mk1WalkingTask(config['task'], 'cuda:0', 0, False)
    
    return task

vecenv.register('RLGPU',
                    lambda config_name, num_actors: RLGPUEnv(config_name))

env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_func,
    })


    
config_path = './cfg/rl_games_walking_train.yaml'
  
# open the config file 
with open(config_path, 'r') as stream:
    try:
        rlg_config_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:    
        raise FileNotFoundError( f"File {config_path} not found")

runner = Runner()
runner.load(rlg_config_dict)
runner.reset()

runner.run({
    'train': True,
    'play': False
})
