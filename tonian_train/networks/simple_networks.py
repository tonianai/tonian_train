

from typing import Callable, Dict, Union, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import gym
import os
import torch.nn as nn
import numpy as np

from tonian_train.common.spaces import MultiSpace
from tonian_train.common.aliases import ActivationFn, InitializerFn


from tonian_train.networks.network_elements import *
from tonian_train.networks.network_elements import ActivationFn, MultiSpace, MultispaceNet, Optional, gym, nn


class A2CBaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None

    def is_dynamics_model(self):
        return False

    def forward(self, actor_obs: Dict[str, torch.Tensor], critic_obs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on a A2C sequential logstd policy

        Args:
            actor_obs (Dict[torch.Tensor]): Multispace Observation of the actor
            critic_obs (Optional[Dict[torch.Tensor]], optional): Multispace Observation of the critic, that are additional to the actor_ibs.
                Defaults to None.
        """
        raise NotImplementedError()


class A2CSimpleNet(A2CBaseNet):

    def __init__(self,
                 action_space: gym.spaces.Space,
                 shared_net: Optional[MultispaceNet] = None,
                 actor_net: Optional[MultispaceNet] = None,
                 critic_net: Optional[MultispaceNet] = None, 
                 action_activation: ActivationFn = nn.Identity(),
                 is_std_fixed: bool = False,
                 std_activation: ActivationFn = nn.Identity(),
                 value_activation: ActivationFn = nn.Identity(),
                 value_size: int = 1,
                 obs_space: MultiSpace = None,
                 has_residual_dynamics: bool = False
                 ) -> None:
        """ A2CPolicy with seperated actor_critic approach and an optional preditcion of the next state

            Note: in this approach the critic also has all the actor observations


                                                                        |---------------------|
                                                                    --> |   res_dynamics_net  | -> next_state
                                                                    |   |---------------------|
                                                                    |
                                                      (optional)
                        |-----------------|         |-------------|     |--------------------|
              obs  ---> |    shared_net   | -->     |  actor_net  | --> | residual_actor_net | -> action_dist -> action
                        |-----------------| \       |-------------|     |--------------------|
                                             \        (optional)
                                              \     |-------------|     |--------------------|
                                               ---> |  critic_net | --> | residual_critic_net| -> value
                                                    |-------------|     |--------------------|


            # If the shared net is not set:
                                            |---------------------|
                                        --> |   res_dynamics_net  | -> next_state
                                        |   |---------------------|
                                        |

                       |------------|         |--------------------|
              obs ->   | actor_net  | -->     | residual_actor_net | -> action_dist -> action
                       |------------|         |--------------------|

                       |------------|         |--------------------|
              obs ->   | critic_net | -->     | residual_critic_net| -> value
                       |------------|         |--------------------|

        Args:
            shared_net (MultispaceNet): Network, that takes in the obs and is a predceding net for both actor and critic
            action_space (gym.spaces.Space): The space the output of the policy should conform to
            action_activation (ActivationFn): The activation function of the actions 
            is_std_fixed (bool): Determines whether the action stanadard deviation (aslo called sigma) is dependend on the output of the aciton
                - it is a parameter of the network eitherways
            critic_net (MultispaceNet): Network of the critic
            residual_actor_net (MultispaceNet): residual network of the actor
            residual_critic_net (MultispaceNet): residual netowrk of the crititic, that takes outputs from shared_actor_net and from critic_net
        """
        super().__init__()
        self.shared_net = shared_net
        self.has_shared_net = shared_net is not None
        self.actor_net = actor_net
        self.has_actor_net = actor_net is not None
        self.critic_net = critic_net
        self.has_critic_net = critic_net is not None
        self.obs_space = obs_space
        self.has_residual_dynamics = has_residual_dynamics
        
        self.residual_dynamics_net = None

        assert self.has_shared_net or (
            self.has_actor_net and self.has_critic_net), 'Either the shared net, or the '
        assert len(
            action_space.shape) == 1, 'Multidim actions are not yet supported'
        self.num_actions = action_space.shape[0]

        if self.has_actor_net:
            self.residual_actor_net = torch.nn.Linear(
                actor_net.out_size(), self.num_actions)
            
            if self.has_residual_dynamics:
                self.residual_dynamics_net = ResidualDynamicsNet( actor_net.out_size(), self.obs_space)
            
        else:
            self.residual_actor_net = torch.nn.Linear(
                shared_net.out_size(), self.num_actions)

            
            if self.has_residual_dynamics:
                self.residual_dynamics_net = ResidualDynamicsNet( actor_net.out_size(), self.obs_space)
                
                
        if self.has_critic_net:
            self.residual_critic_net = torch.nn.Linear(
                critic_net.out_size(), value_size)
        else:
            self.residual_critic_net = torch.nn.Linear(
                shared_net.out_size(), value_size)

        self.action_space = action_space

        self.action_mu_activation = action_activation
        self.value_activation = value_activation

        self.is_continuous = isinstance(self.action_space, gym.spaces.Box)

        self.is_std_fixed = is_std_fixed
        if self.is_std_fixed:
            self.action_std = nn.Parameter(torch.zeros(
                self.num_actions, requires_grad=True))
        else:
            self.action_std = torch.nn.Linear(
                shared_net.out_size(), self.num_actions)

        assert self.is_continuous, "Non continuous action spaces are not yet supported in A2CSequentialPolicyLogStd"

        self.std_activation = std_activation

    def is_rnn(self):
        return False

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass on a A2C sequential logstd policy

        Args:
            actor_obs (Dict[torch.Tensor]): Multispace Observation of the actor
        """

        s_out = obs

        if self.has_shared_net:
            s_out = self.shared_net(s_out)
            s_out = s_out.flatten(1)  # the batch must be preserved

        a_out = s_out
        c_out = s_out

        if self.has_actor_net:
            a_out = self.actor_net(a_out)
            a_out = a_out.flatten(1)  # the batch must be preserved

        if self.has_critic_net:
            c_out = self.critic_net(c_out)
            c_out = c_out.flatten(1)  # the batch must be preserved

        value = self.value_activation(self.residual_critic_net(a_out))

        if self.is_continuous:
            mu = self.action_mu_activation(self.residual_actor_net(a_out))

            if self.is_std_fixed:
                std = self.std_activation(self.action_std)
            else:
                std = self.std_activation(self.action_std(a_out))

            next_state_pred = None
            
            if self.has_residual_dynamics:
                next_state_pred = self.residual_dynamics_net(a_out)

            # mu*0 + std uses the approptiate shape
            return mu, mu*0 + std, value, next_state_pred

        else:
            raise Exception("non continuous spaces are not yet implemented")




def build_simple_a2c_from_config(config: Dict,
                                 obs_space: MultiSpace,
                                 action_space: gym.spaces.Space,
                                 sequence_length: int = -1) -> A2CBaseNet:
    """build the A2C Shared Net Log Std

    Args:
        config (Dict): config
        obs_space (MultiSpace): observations
        action_space (gym.spaces.Space): actions space

    Returns:
        A2CSharedNetLogStd
    """

    actor_net = None
    critic_net = None
    shared_net = None
    
    # We have to expand the obs space, because we have a sequences of observations and this mdoule is not aware of that 
    if sequence_length == -1:
        expanded_sequnced_obs_space = obs_space
    else:
        space_dict = {}
        for key, space in obs_space:
            shape = list(space.shape)
            shape[0] = shape[0] * (sequence_length +1)
            shape = tuple(shape)  
            space_dict[key] = gym.spaces.Box(low= -1, high= 1, shape = shape)
            
        expanded_sequnced_obs_space = MultiSpace(space_dict)

    if "shared_net" in config:
        shared_net = MultiSpaceNetworkConfiguration(
            config['shared_net']).build(expanded_sequnced_obs_space)

    if "actor_net" in config:
        actor_net = MultiSpaceNetworkConfiguration(
            config['actor_net']).build(expanded_sequnced_obs_space)

    if "critic_net" in config:
        critic_net = MultiSpaceNetworkConfiguration(
            config['critic_net']).build(expanded_sequnced_obs_space)

    action_activation = ActivationConfiguration(
        config.get('action_activation', 'None')).build()
    std_activation = ActivationConfiguration(
        config.get('std_activation', 'None')).build()

    value_activation = ActivationConfiguration(
        config.get('value_activation', 'None')).build()

    value_size = config.get('value_size', 1)
    
    has_dynamics_output = config.get('has_dynamics', True)
    

    return A2CSimpleNet(action_space, 
                        shared_net, 
                        actor_net, 
                        critic_net, 
                        action_activation, 
                        True, 
                        std_activation, 
                        value_activation, 
                        value_size, 
                        obs_space, 
                        has_dynamics_output)
