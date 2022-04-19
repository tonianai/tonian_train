

from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.tasks.mk1.mk1_walking.mk1_walking_task import Mk1WalkingTask

from tonian.common.config_utils import task_from_config

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner 
from torch.utils.tensorboard import SummaryWriter
 
import yaml, argparse, os


from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import DefaultAlgoObserver
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

class AlgoObserverNamedRun(DefaultAlgoObserver):
    
    def __init__(self, run_path: str):
        self.run_path = run_path
        super().__init__()
        
    def after_init(self, algo):
        super().after_init(algo)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, "logs"))
    
     

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



if __name__ == '__main__':
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-run_name", required=False, default="", help="Name For the RUN")
    
    args = vars(ap.parse_args())
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

    if args['run_name']:
        rlg_config_dict['params']['config']['full_experiment_name'] = 'mk1_running_'+ args['run_name']
    
    runner = Runner()
    runner.load(rlg_config_dict)
    runner.reset()

    runner.run({
        'train': True,
        'play': False
    })
