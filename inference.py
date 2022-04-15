"""
An Inference run is a run of a previous stored, without training
    
"""
from tonian.tasks.cartpole.cartpole_task import Cartpole

import yaml, os, argparse
from tonian.training.common.logger import TensorboardLogger
from tonian.common.config_utils import  policy_from_config, task_from_config, get_run_index
from tonian.common.utils import set_random_seed

import torch

 

if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=True, help="Name of the environment you want to run : optional run number eg. Cartpole:10")
    ap.add_argument("-at_n_trained_steps", required=False, help="The point of amount of steps at closest to which inference should start", default= None)
    ap.add_argument("-n_steps", required=False, help="The amount of steps the inference is shown", default=1e7)
    
    
    
    args = vars(ap.parse_args())
    n_rollout_steps = args['n_steps']
    env_name = ''
    run_nr = ''
    
    run_base_folder= 'runs/'
    
    if ':' in args['run']:
        # the run nr is set in the run string 
        args_arr = args['run'].split(':')
        
        if len(args_arr) == 2:
              
            env_name = args_arr[0]
            run_nr = args_arr[1]    
            run_folder = os.path.join(run_base_folder, env_name, run_nr)
            
        else:
            
            env_name = args_arr[0]
            batch_name = args_arr[1]
            run_nr = args_arr[2]    
            run_folder = os.path.join(run_base_folder, env_name, batch_name,  run_nr)           
            
    else:
        # the run number is not set in the run string
        # -> use the most recent one
        env_name = args['run']
        run_nr = get_run_index(run_base_folder + env_name) - 1
        
        run_folder = os.path.join(run_base_folder, env_name, run_nr)
    
    if not os.path.exists(run_folder):
        raise FileNotFoundError("The run path does not exist")
    
    device = "cuda:0"
    
    config_path = run_folder + '/config.yaml'
    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {config_path} not found")
    
    if "seed" in config:   
        set_random_seed(config["seed"])
    
    
    task = task_from_config(config['task'])
    task.is_symmetric = False
    
    policy = policy_from_config(config['policy'], task)
    
    # get the highest step stored policy
    
    saves_folder = os.path.join(run_folder, 'saves')
    saves_in_directory = os.listdir(saves_folder)
    
    
    
    if len(saves_in_directory) == 0:
        print("WARNING: The run has no saves policy!!!")
    elif len(saves_in_directory) == 1:
        # only one policy -> use that 
        policy.load(os.path.join(saves_folder, saves_in_directory[0]))
    else:
        
        path_to_best = os.path.join(run_folder, 'saves', 'best.pth')
        policy.load(path_to_best)
        
        #     
        #     if args['at_n_trained_steps'] is None:
        #     # multiple policies -> choose the last or the closest to the goal one    
        #         step_nr = max([int(file_name.split('.')[0]) for file_name in saves_in_directory])
        #         file_name = str(step_nr) + '.pth'
        #     else:
        #         at_n_traines_steps = int(args['at_n_trained_steps'])
        #         
        #         minim_delta = 1e10
        #         file_name_at_min_delta = ''
        #         
        #         for file_name in saves_in_directory:
        #             step_nr = int(file_name.split('.')[0])
        #             
        #             if minim_delta >= abs(step_nr - at_n_traines_steps):
        #                 minim_delta = abs(step_nr - at_n_traines_steps)
        #                 file_name_at_min_delta = file_name 
        #         
        #         if file_name_at_min_delta != '':
        #             file_name = file_name_at_min_delta
        #             
        #     policy.load(os.path.join(saves_folder, file_name))
        #     
        
    
    policy.eval()
    
    last_obs = task.reset()
    
    for i in range(int(n_rollout_steps)):
        with torch.no_grad():
            actions, values, log_probs = policy.forward(last_obs[0], last_obs[1])
    
        new_obs, rewards, dones, info, _ = task.step(actions)
        
        last_obs = new_obs
    
    # get the policy 
    
    print(run_folder)