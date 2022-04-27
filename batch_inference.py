
from tonian.tasks.cartpole.cartpole_task import Cartpole

import yaml, os, argparse, torch
from tonian.common.config_utils import  policy_from_config, task_from_config, get_run_index
from tonian.common.utils import set_random_seed


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-batch", required=True, help="Name of the environment you want to run : batch_id eg. Cartpole:low_cost")


    args = vars(ap.parse_args())
    # n_rollout_steps = args['n_steps']
    env_name = ''
    run_nr = ''
    
    run_base_folder= 'runs/'
    
    if ':' in args['batch']:
        # the run nr is set in the run string 
        args_arr = args['batch'].split(':')
        env_name = args_arr[0]
        batch_id = args_arr[1]
    else:
        raise argparse.ArgumentTypeError("The batch must specify task:batch_id")
    
    batch_base_folder = os.path.join(run_base_folder, env_name, batch_id)
    
    
    if not os.path.exists(batch_base_folder):
        print(f"Batch folder {batch_base_folder}")
        raise FileNotFoundError("The batch path does not exist")
    
    
    run_base_folders = sorted([ os.path.join(batch_base_folder, run_name ) for run_name in os.listdir(batch_base_folder)])
    
    # Trust, that the task is the same across the whole set
    
    config_path = os.path.join(run_base_folders[0], 'config.yaml')
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {config_path} not found")
    
    
    task = task_from_config(config['task'])
    task.is_symmetric = False
    
    for run_base_folder in run_base_folders:
        
        print("########################################################")
        print(f"Running policy from folder: {run_base_folder}")
        
        # get the policy from the specific seed
        
        s_config_path = os.path.join(run_base_folder, 'config.yaml')
        
        # open the config file 
        with open(s_config_path, 'r') as stream:
            try:
                s_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"Algo Config File {s_config_path} not found")
        
        policy = policy_from_config(s_config['policy'], task)
        
        # get the best model from the saves folder
        
        path_to_best = os.path.join(run_base_folder, 'saves', 'best.pth')
        policy.load(path_to_best)
        
        
        
        policy.eval()
        
        last_obs = task.reset()
        
        for _ in range(300):
            
            with torch.no_grad():
                actions, values, log_probs = policy.forward(last_obs[0], last_obs[1])
                
            new_obs, rewards, dones, info, _ = task.step(actions)
            
            last_obs = new_obs
            
        
        
        
        
        
        
        
        
        