
import argparse

from tonian_train.train import train
 
from testing_env.common.config_utils import task_from_config


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-seed", required=False, default = 0, help="Seed for running the env")
    ap.add_argument("-cfg", required= False, default= 'cfg/mk1-simple-seq-test.yaml', help="path to the config")
    ap.add_argument("-batch_id", required= False, default= None,  help="name of the running batch")
    ap.add_argument("-model_out", required=False,default= None, help="The name under wich the model will be registered in the models folder" )
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--no-headless', action='store_false')

    ap.set_defaults(feature= False)
    
    args = vars(ap.parse_args())
    
    train(args['cfg'],task_from_config, args.get('seed', 0), {}, args['headless'], args.get('batch_id', None), args.get('model_out', None),  True)


