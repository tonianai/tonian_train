from abc import ABC, abstractmethod
from typing import Union, Tuple, Any
from common.utils.type_aliases import ContinuousInputSpace, ContinuousOutputSpace, Schedule

import torch as torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from common.utils.utils import get_device

class BasePolicy(ABC, nn.Module):
    
    def __init__(self, input_space: ContinuousInputSpace, output_space: ContinuousOutputSpace, device: str) -> None:
        super().__init__()
        self.input_space = input_space
        self.output_space = output_space
        self.action_size = output_space.action_size
        self.visual_obs_shape = input_space.visual_obs_shape
        self.linear_obs_shape = input_space.linear_obs_size
        self.command_size = input_space.command_size
        
        self.device = device
        
        
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
        
    
    
    

class BaseActorCritic(BasePolicy):
    
    @abstractmethod
    def act(self, obs: torch.Tensor, visual_obs: torch.Tensor, command: torch.Tensor):
        """Generate an action from observation torch Tensors

        Args:
            obs (torch.Tensor): [description]
            visual_obs (torch.Tensor): [description]
            command (torch.Tensor): [description]
        """
        pass
    
    
    @abstractmethod
    def evaluate(self,  obs: torch.Tensor, visual_obs: torch.Tensor, command: torch.Tensor):
        pass
    

class ContVisualComandActorCritic(BasePolicy):
    
    
    def __init__(self, input_space: ContinuousInputSpace, 
                 output_space: ContinuousOutputSpace, 
                 actor_model: nn.Module = None,
                 critic_model : nn.Module = None,
                 conv_model : nn.Module= None,
                 cnn_head_size: int = 0,
                 action_std_init : float = 0.4,
                 device: str = "cpu"
                 ) -> None:
        super().__init__(input_space, output_space, device=device)
        
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.conv_model = conv_model
        
        self.cnn_head_size = cnn_head_size
        
        self.action_std = action_std_init
        
        self.action_var = torch.full((output_space.action_size, ), action_std_init * action_std_init).to(self.device) # Variance vector in the shape of the action size
        
        #self.cnn_head_size = 0 # the size of the flattened tensor that comes out of the conv_model
        
     
            
    def set_action_std(self, new_action_std: float) -> None:
        self.action_std = new_action_std
        self.action_var = torch.full((self.action_size, ), new_action_std * new_action_std).to(self.device)
            
    def act(self, obs: torch.Tensor, visual_obs: torch.Tensor ,command: torch.Tensor):
        """Create an action based on the visual ibs and the linear obs and the command tensor

        Args:
            visual_obs (torch.Tensor): The expected shape of the visual_obs is (batch size, channelsize (default 3), width (84), height(84) ) 
            obs (torch.Tensor): The expected shape of the obs is (batch_size, obs_size)
            command (torch.Tensor):  The expected shape if action is (batch_size, action_size)
        """
        
        if self.input_space.has_visual():
            assert visual_obs.shape[1] == self.input_space.get_channel_size(), "The channel size is not correct, the visual observation may be in the wrong shape correct shape is (batch_size, channel_size, x, y)"
        
        
        batch_size = visual_obs.shape[0] 
        
        cnn_res = self.conv_model(visual_obs)
        cnn_res = torch.reshape(cnn_res, ( batch_size, -1))
        
        #cnn_res = self.conv_model(visual_obs).view(-1, self.cnn_head_size)
        
        action_mean = self.actor_model(torch.cat((cnn_res, obs, command), dim=1)) # concatinate the results of the cnn and the non visual observation vector
        
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)# create the diagonal covariance matrix 
        dist = MultivariateNormal(action_mean, cov_mat)# sample from an normal distribution using the means and the covariance matrix
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
        
    def evaluate(self, obs: torch.Tensor,visual_obs: torch.Tensor,  command: torch.Tensor ,action: torch.Tensor):
        """[summary]Evaluate the obs and the action under a given command
        

        Args:
            visual_obs (torch.Tensor): The expected shape of the visual_obs is (batch size, channelsize (default 3), width (84), height(84) ) 
            obs (torch.Tensor): The expected shape of the obs is (batch_size, obs_size)
            command (torch.Tensor): The expected shape of the command tensor is (batch_size, command_size)
            action (torch.Tensor): The expected shape if action is (batch_size, action_size)
        """
        cnn_res = self.conv_model(visual_obs).view(-1, self.cnn_head_size)
        action_mean = self.actor_model(torch.cat((cnn_res, obs, command), dim=1)) # concatinate the results of the cnn and the non visual observation vector
        
        
        action_var = self.action_var.expand_as(action_mean)
        
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
    
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        cnn_res = self.conv_model(visual_obs).view(-1, self.cnn_head_size)
       
        state_values = self.critic_model(torch.cat((cnn_res, obs, command), dim = 1))
        
        return action_logprobs, state_values, dist_entropy
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_cnn_output_flat_size(conv_model: nn.Module, visual_obs_shape: Union[int, int,int]) -> int:
        """Return the size that a conv net produces as a flattened output

        Args:
            conv_model (nn.Module): Convolutional model used
            visual_obs_shape (Union[int, int,int]): Expected Input Used, channels, x, y 

        Returns:
            int: Size of the expected flattened output for a batch size of 1
        """
        with torch.no_grad():
            # Batch size, color channels, x, y
            test_batch_size = 10
            test = torch.zeros((test_batch_size,) + visual_obs_shape)
            
            result = torch.reshape(conv_model(test),(test_batch_size, -1))
            cnn_head_size = result.shape[1]
        
        return cnn_head_size    
    
        
        
class DefaultActorCritic(ContVisualComandActorCritic):
    
    def __init__(self, input_space: ContinuousInputSpace, output_space: ContinuousOutputSpace, action_std_init: float = 0.4, device: str = "cpu") -> None:
        
        
        if not input_space.has_visual:
            raise Exception("The input space has to include visual fot the default Actor Critic Agent")
        
        
        
        conv_model =  nn.Sequential(
            nn.Conv2d(input_space.visual_obs_shape[0], 6, 4),
            nn.MaxPool2d(4,4),
            nn.Conv2d(6, 16, 4),
            nn.MaxPool2d(3,3),
            nn.Conv2d(16, 30, 4 )
        )
        
        cnn_head_size = ContVisualComandActorCritic.get_cnn_output_flat_size(conv_model, input_space.visual_obs_shape)
        
        linear_start_size = cnn_head_size + input_space.linear_obs_size + input_space.command_size
        
        
        actor_model =  nn.Sequential(
                        nn.Linear(linear_start_size , 64),  
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, output_space.action_size),
                        nn.Tanh()
                    )
        
        critic_model = nn.Sequential(
                        nn.Linear(linear_start_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
        super().__init__(input_space, output_space, actor_model=actor_model, critic_model=critic_model, conv_model=conv_model, cnn_head_size=cnn_head_size, action_std_init=action_std_init, device=device)
    
    
        