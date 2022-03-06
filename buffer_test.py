  
from tonian.common.buffers import DictRolloutBuffer
from tonian.common.spaces import MultiSpace
from gym import spaces
import torch 

n_envs = 100
n_actions = 5
n_obs = 4
batch_size = 64
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
    
    critic_obs = {'linear': obs.clone()}
    actor_obs = {'linear': obs.clone()}
    actions = torch.ones(size = (n_envs, n_actions)) *  torch.arange(start = i * n_envs, end = (i+1) *n_envs).view(n_envs, 1)
    reward = torch.arange(start = i * n_envs, end = (i+1) *n_envs).view(n_envs)
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
 
log_prob_error = False
actor_observation_error = False
critic_observation_error = False
reward_error = False

dones = torch.zeros(size= (n_envs, ))
buffer.compute_returns_and_advantages(values.squeeze(), dones)


for rollout_data in buffer.get(batch_size):   
    
    # measure everything aigainst the action
    print(rollout_data.actions)
    print(rollout_data.old_log_prob)
    if (rollout_data.old_log_prob[0] != rollout_data.actions[0][0] ):
        log_prob_error = True
        
    if(rollout_data.actor_obs["linear"][0][0] != rollout_data.actions[0][0] ):
        actor_observation_error = True
        
    if(rollout_data.critic_obs["linear"][0][0] != rollout_data.actions[0][0] ):
        critic_observation_error = True

if log_prob_error:
    print("Log prob does not correspond to action")
else:
    print("Log prob does correspond to action")
    
    
if actor_observation_error:
    print("Actor obs does not correspond to action")
else:
    print("Actor obs does correspond to action")
    
    
if critic_observation_error:
    print("Critic obs does not correspond to action")
else:
    print("Critic obs does correspond to action")

        