from tonian.policies.policies import SimpleActorCriticPolicy
from tonian.common.buffers import DictRolloutBuffer
from tonian.common.spaces import MultiSpace
from gym import spaces
from tonian.common.schedule import Schedule
import torch 


"""Compare the output and log prob gotten by policy_evaluate_acitons 
    and forward pass 
"""

n_obs = 5
n_actions = 5
batch_size = 10


critic_obs_space = MultiSpace(spaces=  {'linear': spaces.Box(low= -1, high = 1, shape= (n_obs, ))})
actor_obs_space = MultiSpace(spaces =  {'linear': spaces.Box(low= -1, high = 1, shape= (n_obs, ))})

actions_space = spaces.Box(low = -1, high=  1, shape= (n_actions, ))


lr_schedule = Schedule(0.00003)

policy = SimpleActorCriticPolicy(actor_obs_space=actor_obs_space,
                                 critic_obs_space=critic_obs_space,
                                 action_space=actions_space,
                                 lr_schedule=lr_schedule,
                                 init_log_std = 0.0,
                                 actor_hidden_layer_sizes=( 64, 64),
                                 critic_hiddent_layer_sizes=(64, 64),
                                 device = "cpu")

 
dummy_obs_tensor = torch.rand(batch_size, n_obs)
dummy_obs_critic = {"linear": dummy_obs_tensor}
dummy_obs_actor = {"linear": dummy_obs_tensor.clone()}

action, value, log_prob = policy.forward(actor_obs=dummy_obs_actor, critic_obs=dummy_obs_critic)
 


# reshape the actions of and observations
action = action[ 0: int(batch_size/2),: ]
dummy_obs_tensor = dummy_obs_actor['linear'][0: int(batch_size/2), :]
dummy_obs_actor = {'linear': dummy_obs_tensor}

dummy_obs_tensor = dummy_obs_critic['linear'][0: int(batch_size/2), :]
dummy_obs_critic = {'linear': dummy_obs_tensor}

values, log_prob, entropy = policy.evaluate_actions(dummy_obs_actor, dummy_obs_critic, action)




