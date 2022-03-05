from matplotlib.font_manager import is_opentype_cff_font
from tonian.common.buffers import DictRolloutBuffer
from tonian.common.spaces import MultiSpace
from gym import spaces
import torch 

n_envs = 10
n_actions = 5
n_obs = 4
batch_size = 100
buffer_size = 10000

n_steps = int(buffer_size / n_envs)

critic_obs_space = MultiSpace(spaces=  {'linear': spaces.Box(low= 0, high = buffer_size * n_obs, shape= (n_obs, ))})
actor_obs_space = MultiSpace(spaces =  {'linear': spaces.Box(low= 0, high = buffer_size * n_obs, shape= (n_obs, ))})

actions_space = spaces.Box(low = 0, high= buffer_size * n_actions, shape= (n_actions, ))


buffer = DictRolloutBuffer(buffer_size=buffer_size,
                           critic_obs_space=critic_obs_space, 
                           actor_obs_space= actor_obs_space,
                           action_space= actions_space,
                           device= "cpu",
                           n_envs=n_envs)

for i in range(buffer_size):
    
    obs = torch.ones(size = (n_envs, n_obs)) *  torch.arange(start = i * n_envs, end = (i+1) *n_envs).view(n_envs, 1)
    print(obs)
    
    critic_obs = {'linear': obs.clone()}
    actor_obs = {'linear': obs.clone()}
    actions = torch.ones(size = (n_envs, n_actions)) *  torch.arange(start = i * n_envs, end = (i+1) *n_envs).view(n_envs, 1)
    reward = torch.ones(size= (n_envs, ))
    log_probs = torch.arange(start = i * n_envs, end = (i+1) *n_envs).view(n_envs, 1)
    is_episode_start = torch.zeros(size = (n_envs, ))
    values = torch.ones( size = (n_envs , 1))
    
    buffer.add(
        actor_obs=actor_obs,
        critic_obs=critic_obs,
        action = actions,
        reward = reward,
        is_epidsode_start= is_episode_start,
        value = values,
        log_prob=log_probs
        
    )

print(buffer.size())
    
for rollout_data in buffer.get(batch_size): 
    print("log probs")
    print(rollout_data.old_log_prob)
    print("actions")
    print(rollout_data.actions)