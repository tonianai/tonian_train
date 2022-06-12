
from tonian.tasks.cartpole.cartpole_task import Cartpole

import yaml, os, argparse, torch
from tonian.common.config_utils import  task_from_config
from tonian.training.policies import  build_A2CSequentialLogStdPolicy


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
    
    
    for name, param in policy.named_parameters():
        print(name)
        print(param)
        
    
    policy.eval()
    

    last_obs = task.reset()
    
    for _ in range(30000):
        
        with torch.no_grad():
            res = policy.forward( is_train= False, actor_obs= last_obs[0],critic_obs= last_obs[1])
            
        new_obs, rewards, dones, info, _ = task.step(res['actions'])
        
        last_obs = new_obs