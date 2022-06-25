from train import train
import argparse
import torch.multiprocessing as _mp
import torch

mp = _mp.get_context('spawn')

print("Cuda: " + str(torch.cuda.is_available()))

def mp_start_run(queue, done_event):
    args = queue.get()
    train(args[0]['cfg'], 0, {}, True, args[0]['batch_id'], args[0]['model_out'], False, 5e8)
    done_event.set()


if __name__ == '__main__':
    #est_for_values = [15, 30, 40, 50, 60]
    #test_for_dicts = [{'task': {'env' : {'reward_weighting': {'death_cost': value}}}} for value in test_for_values]
    
  
    

    # test_for_values = [0, 0.1, 0.3, 0.5,1, 2, 3]
    # test_for_energy_cost = [0, 0,0, 0]
    # test_for_upright_punishment = []
    
    #test_for_dicts = [{'task': {'env' : {'reward_weighting': rewards }}} for rewards in reward_weightings]
    # test_for_dicts = [{'algo': {'env' : {'reward_weighting': rewards }}} for rewards in reward_weightings]

    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", required= True, help="path to the config")
    ap.add_argument("-batch_id", required= True, help="A name for the run can be set here")
    ap.add_argument("-seed", required=False, help="Seed for running the env") 
    ap.add_argument("-model_out", required=False,default= None, help="The name under wich the model will be registered in the models folder" )
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')
    ap.set_defaults(feature= False)

    args = vars(ap.parse_args()) 



    for i in range(20):

        #print(f"Tesing for value {test_for_values[i]}")
        print(f"Testing for config {str(i)}")
        queue = mp.Queue()
        done_event = mp.Event()
        
        queue.put((args, False, True, 5e8))
        
        #train(args, verbose= False, early_stopping=True, early_stop_patience= 5e7, config_overrides=test_dict)
        p = mp.Process(target=mp_start_run, args=(queue, done_event))
        p.start()
        done_event.wait()


    print("Run complete")

