import time
from warnings import resetwarnings

import numpy as np
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.common.buffers import DictRolloutBuffer
from tonian.policies.policies import SimpleActorCriticPolicy
from tonian.common.spaces import MultiSpace
from tonian.common.schedule import Schedule
from gym import spaces
time.sleep(1.0)
import torch 

n_envs = 10
batch_size = 10
buffer_size = 1000

device = "cuda"

 
env = Cartpole(config_or_path={"env": {"num_envs": n_envs}}, sim_device=device, graphics_device_id=0, headless=False)

env.is_symmetric = False

critic_obs_space = env.critic_observation_spaces
actor_obs_space = env.actor_observation_spaces

actions_space = env.action_space



buffer = DictRolloutBuffer(buffer_size=buffer_size,
                           critic_obs_space=critic_obs_space, 
                           actor_obs_space= actor_obs_space,
                           action_space= actions_space,
                           store_device= "cpu",
                           out_device = device,
                           n_envs=n_envs)

    
lr_schedule = Schedule(0.00003)

policy = SimpleActorCriticPolicy(actor_obs_space=actor_obs_space,
                                 critic_obs_space=critic_obs_space,
                                 action_space=actions_space,
                                 lr_schedule=lr_schedule,
                                 init_log_std = 0.0,
                                 actor_hidden_layer_sizes=( 64, 64),
                                 critic_hiddent_layer_sizes=(64, 64),
                                 device = device)

obs = env.reset()






# fill the rollout buffer 
for i in range(buffer_size):
    
     
    action, value, log_prob = policy.forward(actor_obs=obs[0], critic_obs=obs[1])
    
    print(log_prob)
    
    next_obs, rewards, do_reset, _ = env.step(actions= action)
    
    actor_obs = {'linear': obs[0]['linear'].detach().cpu()}
    crititc_obs = {'linear': obs[1]['linear'].detach().cpu()}

    buffer.add(
        actor_obs=actor_obs,
        critic_obs=crititc_obs,
        action = action.detach().cpu(),
        reward = rewards,
        is_epidsode_start= do_reset,
        value = value,
        log_prob=log_prob
        
    )
    
    
    
    
    
    
    if i == 70:
        # save the logprob of this random point in a numpy array
        np_actor_obs=obs[0]['linear'][0].cpu().detach().numpy()
        np_critic_obs=obs[1]['linear'][0].cpu().detach().numpy()
        np_action = action[0].cpu().detach().numpy()
        np_reward = rewards[0].cpu().detach().numpy()
        np_is_epidsode_start= do_reset[0].cpu().detach().numpy()
        np_value = value[0].cpu().detach().numpy()
        np_log_prob=log_prob[0].cpu().detach().numpy()
        
        
        actor_obs = {'linear': obs[0]['linear'].detach().to(device)}
        crititc_obs = {'linear': obs[1]['linear'].detach().to(device)}
        
        actor_obs_1 = actor_obs['linear']
        
        values_old_1, log_prob_old_1, entropy_old_1 = policy.evaluate_actions(actor_obs, crititc_obs, action.to(device))
        
        values_old_1, log_prob_old_2, entropy_old_1 = policy.evaluate_actions(actor_obs, crititc_obs, action.to(device))
        
        values_old_1, log_prob_old_3, entropy_old_1 = policy.evaluate_actions(actor_obs, crititc_obs, action.to(device))
        
        log_prob_old_1 = log_prob_old_1[0]

    
    obs = next_obs
        
print(np_actor_obs)
 
log_prob_error = False
actor_observation_error = False
critic_observation_error = False
reward_error = False

dones = torch.zeros(size= (n_envs, ))
values = torch.ones( size = (n_envs , 1))
buffer.compute_returns_and_advantages(values.squeeze(), dones)


# fake training step -> check if the log prob is still the same as the taken
for rollout_data in buffer.get(batch_size):   
    
    actions = rollout_data.actions
    

                
                
    values, log_prob, entropy = policy.evaluate_actions(rollout_data.actor_obs, rollout_data.critic_obs, actions)
    
    for i in range(batch_size):
        
        #print(actions.shape)
        
        
        if np.array_equal(np_action, actions[i].cpu().detach().numpy()):
            print("Found the action")
            
            print("action")
            print(np_action)
            print(actions[i].detach().cpu().numpy())
            
            
            print("log prob")
            print(f"old {rollout_data.old_log_prob[i].cpu().numpy()}")
            print(f"old_numpy {np_log_prob}")
            print(f"old eval numpy 1 {log_prob_old_1.detach().cpu().numpy()}")
            print(f"new {log_prob[i].detach().cpu().numpy()}")
            
            
            print("Observations")
            print(f"actor old {rollout_data.actor_obs['linear'].detach().cpu().numpy()[i]}")
            print(f"actor old numpy {np_actor_obs}")
            
            
            print("Observations")
            print(f"critic old {rollout_data.critic_obs['linear'].detach().cpu().numpy()[i]}")
            print(f"critic old numpy {np_critic_obs}")
            
            break;
    
    #print("log_prog")
    #print(log_prob)
    #print("old log prob")
    #print(rollout_data.old_log_prob)