# Show all from all the runs in a batch the best policy for a couple seconds
 
from tonian.tasks.cartpole.cartpole_task import Cartpole

import yaml, os, argparse
import argparse
import torch.multiprocessing as _mp
import torch
from tonian.common.utils.config_utils import  policy_from_config, task_from_config, get_run_index
from tonian.common.utils.utils import set_random_seed
#

mp = _mp.get_context('spawn')

def inference(run_folder: str):
    
    print(f"in sec f{run_folder}")
    
    print("-------------------------")
    
    
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
    
    task = task_from_config(config['task'], False)
    task.is_symmetric = False
    
    policy = policy_from_config(config['policy'], task)
    
    # get the highest step stored policy
    
    saves_folder = os.path.join(run_folder, 'saves')
    
    best_model_path = os.path.join(saves_folder, 'best.pth')
    
    policy.load(best_model_path)
    
    
    print("-------------------------")
    
    policy.eval()
    
    last_obs = task.reset()
    
    for i in range(int(10000)):
        with torch.no_grad():
            actions, values, log_probs = policy.forward(last_obs[0], last_obs[1])
    
        new_obs, rewards, dones, info = task.step(actions)
        
        last_obs = new_obs
        
        

def mp_start_inference(queue, done_event):
    args = queue.get()
    
    inference(args[0])
    done_event.set()
    





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
        raise FileNotFoundError("The batch path does not exist")
    
    
    run_base_folders = sorted([ os.path.join(batch_base_folder, run_name ) for run_name in os.listdir(batch_base_folder)])
    
    
    for run_base_folder in run_base_folders:
        
        print("RUN "+ run_base_folder )        
        queue = mp.Queue()
        done_event = mp.Event()
        p = mp.Process(target=mp_start_inference, args=(queue,done_event))
        p.start()
        done_event.wait()
        #input("Hit enter to continue:")
        #p.terminate()