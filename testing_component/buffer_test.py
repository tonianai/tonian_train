from tonian_train.algorithms.transformer_algorithm import SequenceBuffer
import numpy as np
import torch 
from gym.spaces import Box
import gym, math

from tonian_train.common.spaces import MultiSpace



action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

obs_space = MultiSpace({
  "linear":  Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
  "extra": Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
})


print(obs_space.sample())
print(action_space.sample())

horizon_length = 100
seq_len = 50
num_envs = 10

buffer = SequenceBuffer(horizon_length=horizon_length, obs_space=obs_space, action_space= action_space, store_device='cuda:0', out_device= 'cuda_0', n_envs=num_envs)

for i in range(horizon_length):
    
    obs = {
        'linear': torch.zeros((num_envs, 2) , device='cuda:0' ),
        'extra': torch.zeros((num_envs, 2), device='cuda:0')
        
    }
    
    actions_mu = torch.zeros((num_envs, 2), device='cuda:0')
    
    actions_std = torch.zeros((num_envs, 2), device='cuda:0')
    
    values = torch.zeros((num_envs,1 ), device='cuda:0')
    
    dones = torch.zeros((num_envs,), device= 'cuda:0' )
    
    
    for o in range(num_envs):
        obs['linear'][o] = (obs_space.sample()['linear']).to('cuda:0')
        obs['extra'][o] = (obs_space.sample()['extra']).to('cuda:0')
        
        actions_mu[o] = torch.from_numpy(action_space.sample()* 0 + o).to('cuda:0')
        actions_std[o] = torch.from_numpy(action_space.sample()* 0 + 1).to('cuda:0')
        
        values[o] = o + horizon_length
        
        if i == 10:
            dones[0] = 1
            
            
        if i == 11:
            dones[1] = 1
        
        
    buffer.add(obs=obs, action_mu=actions_mu, action_std= actions_std, values= values, dones= dones )
        
        
obs = {
        'linear': torch.zeros((num_envs, 2) , device='cuda:0' ),
        'extra': torch.zeros((num_envs, 2), device='cuda:0')
        
}
    
res = buffer.get_and_merge_last_obs(obs, 10 )

print(res)
