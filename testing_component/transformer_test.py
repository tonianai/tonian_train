
from tonian_train.networks import TransformerNetLogStd, build_transformer_a2c_from_config
import numpy as np
import torch 
from gym.spaces import Box
import gym, math

from tonian_train.common.spaces import MultiSpace

config = {
     "policy": {
       "seq_len": 100,
          
       "network": {
          "d_model": 512,
          "n_heads": 16,
          "num_encoder_layers": 7,
          "num_decoder_layers": 10,
          "is_std_fixed": False,
          
          "policy_type": "a2c_transformer", 
          "normalize_input": True,
          
          "input_embedding": {
            
            "encoder":{
              "network":[
                {
                  "name": "general_linear",
                  "input": [ 
                    {'obs': 'linear'},
                    {'obs': 'extra'}  
                  ],
                  "mlp": {
                     'units': [256, 128],
                     'activation': "RELU",
                     'initializer': "default" 
                  }
                }                
              ]
            }

            
          },
          
          "output_embedding": {
            "encoder": {
               'mlp': {
                  'units': [256, 128],
                  'activation': 'relu',
                  'initializer': 'default' 
                 
               }
              
            }
            
            
          }
            
            
       }
      }
}

action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

obs_space = MultiSpace({
  "linear":  Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
  "extra": Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
})


transformer = build_transformer_a2c_from_config(config=config['policy']['network'], 
                                                obs_space= obs_space,
                                                action_space= action_space,
                                                value_size= 1,
                                                seq_len= 512)




