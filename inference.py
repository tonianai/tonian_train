"""
An Inference run is a run of a previous stored 
    
"""

import yaml, os, argparse
from tonian.common.utils.config_utils import task_from_config


      
def get_run_index(base_folder_name: str) -> int:
    """get the index of the run
    Args:
        base_folder_name (str): The base folder all the runs are stored in 
    """
    if not os.path.exists(base_folder_name):
        raise FileNotFoundError()
        
    n_folders_in_base = len(os.listdir(base_folder_name))
    
    return n_folders_in_base


if __name__ == '__main__':    
    ap = argparse.ArgumentParser()
    ap.add_argument("-run", required=True, help="Name of the environment you want to run : optional run number eg. Cartpole:10")
    
    args = vars(ap.parse_args())
    
    env_name = ''
    run_nr = ''
    
    run_base_folder= 'runs/'
    
    if ':' in args['run']:
        # the run nr is set in the run string 
        args_arr = args['run'].split(':')
        env_name = args_arr[0]
        run_nr = args_arr[1]
    else:
        # the run number is not set in the run string
        # -> use the most recent one
        env_name = args['run']
        run_nr = get_run_index(run_base_folder + env_name) - 1
        
    run_folder = run_base_folder + env_name + '/'+ str(run_nr)
    
    if not os.path.exists(run_folder):
        raise FileNotFoundError("The run path does not exist")
    
    device = "cuda:0"
    
    algo_cofig_path = run_folder + '/algo_config.yaml'
    env_cofig_path = run_folder + '/algo_config.yaml'
    
    # open the config file 
    with open(algo_cofig_path, 'r') as stream:
        try:
            algo_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {algo_cofig_path} not found")
    
    
    # open the config file 
    with open(env_cofig_path, 'r') as stream:
        try:
            env_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"Algo Config File {algo_cofig_path} not found")
    
    
    task = task_from_config(env_config)
    task.is_symmetric = False
    
    # get the policy 
    
    print(run_folder)