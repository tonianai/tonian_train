
from typing import Optional, Union, List, Tuple
import numpy as np
import yaml


import os 
class Command:
    
    
    def __init__(self, config_file: str = './tasks/base/command.yaml', one_hot: bool = True) -> None:
        
        
        
        print(os.getcwd())
        self.mode_amount = 10
        
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:    
                raise FileNotFoundError( f"File {config_file} not found")
              
        command_size = 0 
        
        self.command_dict = {}
        
        # figure out the command size 
        if one_hot:
            
            for mode in config:
                self.command_dict[mode] = command_size
                command_size += 1
                if config[mode]:
                    for sub_mode in config[mode]:
                        self.command_dict[mode + sub_mode] = command_size 
                        command_size += 1
        
        self.command_size = command_size

        
        
    
    def get_one_hot_command(self, mode: str, sub_mode: str = ''):
        
        
        command_int_value = self.command_dict[mode + sub_mode]
        
        if command_int_value == 0:
            return np.zeros(self.command_size -1 )        
        return self.int_to_one_hot( command_int_value - 1)
        

    def int_to_one_hot( self ,value: int):
        zeros = np.zeros(self.command_size -1 , dtype= np.float32)
        zeros[value] = 1.0
        return zeros
    
     
    
    