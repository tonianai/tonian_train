import time
from warnings import resetwarnings

from numpy import roll
from tonian.tasks.cartpole.cartpole_task import Cartpole
from tonian.common.buffers import DictRolloutBuffer
from tonian.policies.policies import SimpleActorCriticPolicy
from tonian.common.spaces import MultiSpace
from tonian.common.schedule import Schedule
from gym import spaces
time.sleep(1.0)
import torch 

n_envs = 100
batch_size = 64
buffer_size = 1000

n_steps = int(buffer_size / n_envs)
 
env = Cartpole(config_or_path={"env": {"num_envs": n_envs}}, sim_device="cpu", graphics_device_id=0, headless=False)

env.is_symmetric = False

critic_obs_space = env.critic_observation_spaces
actor_obs_space = env.actor_observation_spaces

actions_space = env.action_space



buffer = DictRolloutBuffer(buffer_size=buffer_size,
                           critic_obs_space=critic_obs_space, 
                           actor_obs_space= actor_obs_space,
                           action_space= actions_space,
                           device= "cpu",
                           n_envs=n_envs)

    
lr_schedule = Schedule(0.00003)

policy = SimpleActorCriticPolicy(actor_obs_space=actor_obs_space,
                                 critic_obs_space=critic_obs_space,
                                 action_space=actions_space,
                                 lr_schedule=lr_schedule,
                                 init_log_std = 0.0,
                                 actor_hidden_layer_sizes=( 64, 64),
                                 critic_hiddent_layer_sizes=(64, 64),
                                 device = "cpu")

obs = env.reset()

# fill the rollout buffer 
for i in range(buffer_size):
    
     
    action, value, log_prob = policy.forward(actor_obs=obs[0], critic_obs=obs[1])
    
    next_obs, rewards, do_reset, _ = env.step(actions= action)

    buffer.add(
        actor_obs=obs[0],
        critic_obs=obs[1],
        action = action,
        reward = rewards,
        is_epidsode_start= do_reset,
        value = value,
        log_prob=log_prob
        
    )
    
    obs = next_obs
 
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
    
    print("log_prog")
    print(log_prob)
    print("old log prob")
    print(rollout_data.old_log_prob)