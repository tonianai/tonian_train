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
    
    
    multispace_net = net_config.build(multispace).to('cuda:0')
    
    batch_size = 256
    
    linear_samples = []
    command_samples = []
    visual_samples = []
    

    
    for _ in range(batch_size):
        linear_samples.append(torch.from_numpy(linear.sample()).unsqueeze(dim=0).to('cuda:0'))
        command_samples.append(torch.from_numpy(command.sample()).unsqueeze(dim=0).to('cuda:0'))
        visual_samples.append(torch.from_numpy(visual.sample()).unsqueeze(dim=0).to('cuda:0'))
         
    linear_tensor = torch.Tensor((batch_size, 100 )).to('cuda:0')
    command_tensor = torch.Tensor((batch_size,  10) ).to('cuda:0')
    visual_tensor = torch.Tensor((batch_size, 1, 50, 50) ).to('cuda:0')
    print(visual_tensor.shape)
    
    torch.cat(linear_samples, out= linear_tensor)
    torch.cat(command_samples, out= command_tensor)
    torch.cat(visual_samples, out= visual_tensor)
    visual_tensor = visual_tensor.unsqueeze(dim=1)
    print(visual_tensor.shape)
         
    sample_obs = {'linear': linear_tensor,
                  'command': command_tensor,
                  'visual': visual_tensor
                  }
    pred = multispace_net(sample_obs) 
    print(pred)
    
    
    optim = torch.optim.Adam(params=multispace_net.parameters())
    
    for i in range(10):
     
        ones_result = torch.ones(multispace_net(sample_obs).shape, device= "cuda:0")
        pred = multispace_net(sample_obs) 
        loss_fn = torch.nn.MSELoss()
    
        loss = loss_fn(pred, ones_result)
        
        multispace_net.zero_grad()
        loss.backward()
    
        optim.step()
    
    print('--------------------------------------------------------------------------------------------------------')
    
    print(multispace_net.out_size())
    print(multispace_net(sample_obs).shape)
    print(multispace_net(sample_obs))
    
    
    