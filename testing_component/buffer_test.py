from tonian_train.algorithms.sequential_ppo import SequenceBuffer, SequenceDataset
import numpy as np
import torch 
from gym.spaces import Box
import gym, math

from tonian_train.common.spaces import MultiSpace



action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

obs_space = MultiSpace({
  "linear":  Box(low=np.array([-1.0, -1.0, -1,-1,-1]), high=np.array([1.0, 1.0,1,1,1]), dtype=np.float32)
})


print(obs_space.sample())
print(action_space.sample())

horizon_length = 100
seq_len = 10
seq_len = 50
num_envs = 13

buffer = SequenceBuffer(horizon_length=horizon_length, sequence_length= seq_len ,obs_space=obs_space, action_space= action_space, store_device='cuda:0', out_device= 'cuda_0', n_envs=num_envs)

for i in range(horizon_length):
    
    obs = {
        'linear': torch.zeros((num_envs, 5) , device='cuda:0' )
        
    }
    
    actions_mu = torch.zeros((num_envs, 2), device='cuda:0')
    
    actions = torch.zeros((num_envs, 2), device='cuda:0')
    
    actions_std = torch.zeros((num_envs, 2), device='cuda:0')
    
    # EVERY ENV GETS ITS OWN INDEX
    values = torch.arange(num_envs).view((num_envs, 1)).to(device='cuda:0').to(torch.float32)
    # WITHIN THAT INDEX, THE FRACTIONAL VALUES INCREASE
    values += torch.ones_like(values) * i/horizon_length
    
    
    dones = torch.zeros((num_envs,), device= 'cuda:0' )
    
    rewards = torch.zeros((num_envs, 1), device= 'cuda:0' )
    neglogprobs = torch.zeros((num_envs,), device= 'cuda:0' )
     
    
    
    for o in range(num_envs):
        obs['linear'][o] =  torch.arange(i , (i+1), step=0.2).to('cuda:0')  #(obs_space.sample()['linear']).to('cuda:0') 
        
        actions_mu[o] = torch.from_numpy(action_space.sample()* 0 + o).to('cuda:0')
        actions_std[o] = torch.from_numpy(action_space.sample()* 0 + 1).to('cuda:0')
        
        
        if i == 10:
            dones[0] = 1
            
            
        if i == 11:
            dones[1] = 1
        
        
    buffer.add(obs=obs, action_mu=actions_mu, action_std= actions_std, values= values, dones= dones, rewards= rewards, neglogprobs= neglogprobs, action= actions)
        
        
obs = {
        'linear': torch.zeros((num_envs, 5) , device='cuda:0' )
        
}

#  num_envs, seq_pos,  obs_len 
pass

mini_batch_size = 10

dataset = SequenceDataset(buffer, mini_batch_size)

obs =  dataset[i]['obs']['linear']

obs =  dataset[i]['dones']

# what would i expect?
# batch_size, seq_len, obs_len
print(obs.shape)

pass

