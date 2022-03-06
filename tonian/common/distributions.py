"""
    Probability distributions based on the implementation of stable baselines3
"""
    
    
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Optional, Tuple
from cv2 import determinant

import gym
from gym import spaces

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Bernoulli

from tonian.common.schedule import Schedule

class Distribution(ABC):
    """ Abstract Base class for all distributions"""   
    
    def __init__(self) -> None:
        super().__init__()
        self.dist = None
    
    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()
    
    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """
        
def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """ Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
        
    else:
        print("Go to distributions tensors and check if sum_independent is desired")
        tensor = tensor.sum()
        
    return tensor
        
class DiagGaussianDistributionStdParam(Distribution):
    """Gaussioan distribution with diagonal covariance matrice, for continoous acitons
    The std is a nn Param in this implementation
    Args:
        Distribution ([type]): [description]
    """
    
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size
        self.mean_actions = None
        self.log_std = None
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_size)
        log_std = nn.Parameter(torch.ones(self.action_size) * log_std_init, requires_grad=True)
        return mean_actions, log_std
    
    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "DiagGaussianDistributionStdParam":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        #TODO: Change back
        #print("means actions internal")
        #print(mean_actions)
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        #action_std = torch.ones_like(mean_actions) * 0.1
        self.distribution = Normal(mean_actions, action_std)
        return self
        
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        #print("actions internal")
        #print(actions)
        log_prob = self.distribution.log_prob(actions)
        #print("log_prob internal")
        #print(log_prob)
        return sum_independent_dims(log_prob)
    
    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())
    
    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    
class DiagnoalGaussianDistributionStdSchedule(Distribution):
    """Gaussioan distribution with diagonal covariance matrice, for continoous acitons
    The action std is defined by a Schedule in this implementation
    Args:
        Distribution ([type]): [description]
    """
    
    def __init__(self, action_size: int, std_schedule: Schedule) -> None:
        self.action_size = action_size
        self.mean_actions = None
        self.std_schedule: Schedule = std_schedule
        super().__init__()
    
    def proba_distribution_net(self, latent_dim: int) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_size) 
        return mean_actions
    
    
    def proba_distribution(self, mean_actions: torch.Tensor, steps: int) -> "Distribution":
        """
        Create the distribution given its parameters 
        Args:
            mean_actions (torch.Tensor): mean action
            steps (int): steps taken up to this point, used in the scheduled std

        Returns:
        """
        
        action_std = torch.ones_like(mean_actions) * self.std_schedule(steps)
        
        self.distribution = Normal(mean_actions, action_std)
        return self
    

    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())
    
    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean
    
    def actions_from_params(self, mean_actions: torch.Tensor, steps_taken: int, deterministic: bool) -> torch.Tensor:
        """_summary_

        Args:
            mean_actions (torch.Tensor): _description_
            steps_taken (int): _description_
            deterministic (bool): _description_

        Returns:
            torch.Tensor: _description_
        """
        self.proba_distribution(mean_actions, steps_taken)
        return self.get_actions(deterministic= deterministic)
    
        
    
    def log_prob_from_params(self, mean_action: torch.Tensor, steps_taken: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        actions = self.actions_from_params(mean_action, steps_taken)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    
    
    
# todo add more distributions if need be