from train import train
import argparse

test_for_values = [4,8,10,16,20,30,40]
test_for_dicts = [{'task': {'reward_weighting': {'upright_punishment_factor': value} }} for value in test_for_values]

 
ap = argparse.ArgumentParser()
ap.add_argument("-seed", required=False, help="Seed for running the env")
ap.add_argument("-cfg", required= True, help="path to the config")
ap.add_argument('--headless', action='store_true')
ap.add_argument('--no-headless', action='store_false')
ap.set_defaults(feature= False)
    
args = vars(ap.parse_args()) 
    


for i, test_dict in enumerate(test_for_dicts):
    
    print(f"Tesing for value {test_for_values[i]}")
    
    train(args, verbose= False, early_stopping=True, early_stop_patience= 5e7, config_overrides=test_dict)
    
    
print("Run complete")

