from tonian.training2.common.configuration_types import MultiSpaceNetworkConfiguration
from tonian.common.spaces import MultiSpace

import yaml, gym, torch


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
    
    
    multispace_net = net_config.build(multispace)
    
    sample_obs = {'linear': torch.from_numpy(linear.sample()).unsqueeze(dim=0),
                  'command': torch.from_numpy(command.sample()).unsqueeze(dim=0),
                  'visual': torch.from_numpy(visual.sample()).unsqueeze(dim=0).unsqueeze(dim=0)
                  }
    
    print(multispace_net(sample_obs))