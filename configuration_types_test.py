from tonian.training2.common.configuration_types import MultiSpaceNetworkConfiguration
from tonian.common.spaces import MultiSpace

import yaml, gym


if __name__ == '__main__':
    config_path = './tests/test_net_config.yaml'    
    # open the config file 
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:    
            raise FileNotFoundError( f"File {config_path} not found")
        
        
    net_config = MultiSpaceNetworkConfiguration(config)
    
    linear = gym.spaces.Box(low = 0, high = 1, shape = (100,))
    
    command = gym.spaces.Box(low = 0, high = 1, shape = (10,))
    
    visual = gym.spaces.Box(low = 0, high = 1, shape = (50, 50))
    
    multispace = MultiSpace({'linear': linear,
                              'command': command,
                              'visual': visual  
                             })
    
    
    net_config.build(multispace)