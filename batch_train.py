from train import train
import argparse
import torch.multiprocessing as _mp
import torch

mp = _mp.get_context('spawn')

print("Cuda: " + str(torch.cuda.is_available()))

def mp_start_run(queue, done_event):
    args = queue.get()
    train(args[0], args[1], args[2], args[3], args[4], args[0]['batch_id'])
    done_event.set()


if __name__ == '__main__':
    test_for_values = [15, 30, 40, 50, 60]
    test_for_dicts = [{'task': {'env' : {'reward_weighting': {'death_cost': value}}}} for value in test_for_values]

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg", required= True, help="path to the config")
    ap.add_argument("-batch_id", required= True, help="A name for the run can be set here")
    ap.add_argument("-seed", required=False, help="Seed for running the env")
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')
    ap.set_defaults(feature= False)

    args = vars(ap.parse_args()) 



    for i, test_dict in enumerate(test_for_dicts):

        print(f"Tesing for value {test_for_values[i]}")
        
        queue = mp.Queue()
        done_event = mp.Event()
        
        queue.put((args, False, True, 5e7, test_dict))
        
        #train(args, verbose= False, early_stopping=True, early_stop_patience= 5e7, config_overrides=test_dict)
        p = mp.Process(target=mp_start_run, args=(queue, done_event))
        p.start()
        done_event.wait()


    print("Run complete")

