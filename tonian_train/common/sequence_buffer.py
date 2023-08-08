
from typing import Dict, Tuple, Any, Union, Optional, List

from tonian_train.common.spaces import MultiSpace  

import torch.nn as nn
import torch, gym, os, yaml, time

from torch.utils.data import Dataset
from tonian_train.common.running_mean_std import RunningMeanStd, RunningMeanStdObs


class SequenceBuffer():
    
    def __init__(self,
                 n_envs: int,  
                 horizon_length: int,
                 sequence_length: int, 
                 obs_space: MultiSpace,
                 action_space: gym.spaces.Space,
                 store_device: Union[str, torch.device] = "cuda:0",
                 out_device: Union[str, torch.device] = "cuda:0",
                 n_values: int = 1) -> None:
        """Buffer, that stores the observations and the outputs for the transformer sequence length
        
        TODO: fix that horizon length must be bigger than sequence length
        
        Args:
        
            horizon_length (int): the amount of steps taken in one epoch 
            sequence_length (int): the amount of steps the actor can look back
            obs_space (MultiSpace): the observation space 
            action_space (gym.spaces.Space): the action space
            store_device (_type_, optional): the device on which the tensors will be stored. Defaults to "cuda:0".
            out_device (_type_, optional): the device on which the tensors will be outputed. Defaults to "cuda:0".
            n_envs (int, optional): The amount of environments. Defaults to 1. 
            n_values (int, optional): The amount of values. Defaults to 1.
        """
        self.action_space = action_space
        self.action_size = action_space.shape[0]
        assert self.action_size, "Action size must not be zero"
        

        self.n_values = n_values 
        self.horizon_length = horizon_length # The amount of steps played per epoch
        self.sequence_length = sequence_length # The amount of observations and actions taken into account by the network
        
        self.buffer_length = self.horizon_length + self.sequence_length # This buffer length is required, because 
        
        
        self.obs_space = obs_space
        
        self.store_device = store_device
        self.out_device = out_device
        self.n_envs = n_envs

        # The Sequence buffer can go in a state of refusing writes during learning        
        self.can_write = True
        
        
        # the data in these tensors comes from the outside
        self.external_tensor_names = ['dones', 'values', 'rewards', 'action', 'action_mu', 'action_std', 'neglogprobs', 'obs']
        
        # the data from these tensors is derived within the buffer
        self.derived_tensor_names = ['src_key_padding_mask', 'tgt_key_padding_mask', 'returns', 'advantages']
        
        
        self.tensor_names = self.external_tensor_names + self.derived_tensor_names
        
        # ----- Create the buffers and set all initial values to zero
 
        self.dones = torch.zeros(( self.n_envs, self.buffer_length,), dtype=torch.int8, device=self.store_device)
        self.values = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        
        self.rewards = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
     
        self.action = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype=torch.float32, device=self.store_device)
        
        # the mean of the action distributions
        self.action_mu = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype=torch.float32, device=self.store_device)
        # the std(sigma) of the action distributions   
        self.action_std = torch.zeros((self.n_envs, self.buffer_length, self.action_size), dtype= torch.float32, device=self.store_device)
         
        self.neglogprobs = torch.zeros((self.n_envs, self.buffer_length), dtype= torch.float32, device=self.store_device)
     
        self.advantages = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        self.returns = torch.zeros((self.n_envs, self.buffer_length, self.n_values), dtype=torch.float32, device=self.store_device)
        
         
        
        self.obs = {}
        for key, obs_shape in self.obs_space.dict_shape.items():
            self.obs[key] = torch.zeros((self.n_envs, self.buffer_length) + obs_shape, dtype=torch.float32, device= self.store_device)
        
        # when the padding masks are true, the values will be disgarded
        self.src_key_padding_mask = torch.ones((self.n_envs, self.buffer_length), device= self.store_device, dtype=torch.bool)
        self.tgt_key_padding_mask = torch.ones((self.n_envs, self.buffer_length), device= self.store_device, dtype=torch.bool)
          
        
        # the advantage pointer always refer to the second (1) dimesnion, that correspons to the buffer length
        self.left_advantage_pointer = self.buffer_length -1 # the pointer at which the first correct advantages are
        self.right_advantage_pointer = self.buffer_length -1 # the pointer at which the last correct advantages are
  
    def add(
        self, 
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        action_mu: torch.Tensor,
        action_std: torch.Tensor,
        rewards: torch.Tensor, 
        values: torch.Tensor,
        dones: torch.Tensor,
        neglogprobs: torch.Tensor):
        """Add one set of collected experience to the sequence buffer

        Args:
            obs (Dict[str, torch.Tensor]): Observations of the last step shape(num_envs, ) + obs_shape
            action_mu (torch.Tensor): mean of the current actions (num_envs, num_actions) 
            action_std (torch.Tensor): standard deviation of the current actions (num_envs, num_actions) 
            values (torch.Tensor): predicted value of the state (num_envs, num_values)
            dones (torch.Tensor): determines, whether the step was terminal for the episode (num_envs)
        """
        
        assert self.can_write, "The Sequence buffer cannot be written to while the can_write flag is set to false"
        
        # roll the last to the first position
        
        # The tensord fill upd from 
        for key in self.obs:   
            self.obs[key] =  torch.roll(self.obs[key], shifts=(-1), dims=(1)) 
        
        self.action = torch.roll(self.action, shifts=(-1), dims=(1))
        self.action_mu = torch.roll(self.action_mu, shifts=(-1), dims=(1))
        self.action_std = torch.roll(self.action_std, shifts=(-1), dims=(1))
        self.values = torch.roll(self.values, shifts=(-1), dims=(1))
        self.dones = torch.roll(self.dones, shifts=(-1), dims=(1))
        self.neglogprobs = torch.roll(self.neglogprobs, shifts=(-1), dims = (1))
        self.rewards = torch.roll(self.rewards, shifts=(-1), dims=(1))
        
        # advantage and returns will also be rolled and set to 0, so that they align with the rest of the data
        self.advantages = torch.roll(self.advantages, shifts=(-1), dims=(1))
        self.returns = torch.roll(self.returns, shifts=(-1), dims=(1))
        
        self.advantages[:, -1,:] = torch.zeros_like(values, device= self.store_device)
        self.returns[:, -1,:] = torch.zeros_like(values, device= self.store_device)
        
        # both pointers get deducted one, because no additional data was added, the data was just shifted down by one
        # pointers can also not be smaller than 0
        self.left_advantage_pointer -= 1
        if self.left_advantage_pointer <= 0:
            self.left_advantage_pointer = 0
        self.right_advantage_pointer -= 1 
        if self.right_advantage_pointer <= 0:
            self.right_advantage_pointer = 0
        
        
        self.src_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        self.tgt_key_padding_mask = torch.roll(self.tgt_key_padding_mask, shifts = (-1), dims= (1))
        
        for key in self.obs:   
            self.obs[key][:, -1, :] = obs[key].detach().to(self.store_device)
             
            
        self.action[:, -1, :] = action.detach().to(self.store_device)
        self.action_mu[:, -1, :] = action_mu.detach().to(self.store_device)
        self.action_std[:, -1, :] = action_std.detach().to(self.store_device)
        self.values[:, -1,:] = values.detach().to(self.store_device)
        self.dones[:, -1] = dones.detach().to(self.store_device)
        self.neglogprobs[:, -1] = neglogprobs.detach().to(self.store_device)
        self.rewards[:, -1, : ] = rewards.detach().to(self.store_device)
        
        # for every true dones at index 1 -> erase all old states to the left
        # and set the src and tgt key padding masks correctly
        last_dones = self.dones[:, -2].to(torch.bool)
        self.src_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        self.tgt_key_padding_mask[:,-1] = torch.zeros_like(last_dones).to(torch.bool)
        
        # only full row paddings have to be made, because the old padding still exist  
        additive_padding_dones_mask =  torch.unsqueeze((last_dones), 1).tile(1, self.buffer_length)
        additive_padding_dones_mask[:, -1] = torch.zeros_like(last_dones).to(torch.bool)
        
        self.src_key_padding_mask = torch.logical_or(self.src_key_padding_mask, additive_padding_dones_mask)
        self.tgt_key_padding_mask = torch.logical_or(self.tgt_key_padding_mask, additive_padding_dones_mask)

        
    def get_and_merge_last_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Gets the sequence length -1 amount of previous observations and merges them with the obs dict, given as input
        
        This function does not change the obs buffer and is meant as a direct output for the trnsformer, hence the output in the shape of the sequence length
        Args:
            obs_dict (Dict[str, torch.Tensor]): observations of the last world iteration shape(num_envs, ) + obs _shape
    
        Returns:
            Dict[str, torch.Tensor]: dim of output tensors => (n_envs,sequence_length, ) + obs_shape

        """
        
        for key in obs_dict.keys():
            res = self.obs[key][:, -(self.sequence_length-1)::, :] 
            obs_dict[key] = torch.concat((res, torch.unsqueeze(obs_dict[key], 1)), dim= 1)
        
        return obs_dict
    
    
    def get_last_sequence_step_data(self, obs: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """returns the relevant data for the next step 

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]: Dict containing relevant info 
        """
        obs_dict = {}
        for key in self.obs.keys():
            res = self.obs[key][:, -(self.sequence_length)::, :]  
            obs_dict[key] = torch.concat((res, torch.unsqueeze(obs[key], 1)), dim= 1)
            
        sequence_action_mu = self.action_mu[:, -(self.sequence_length)::,:]
        sequence_action_std = self.action_std[:, -(self.sequence_length)::,:]
        sequence_value = self.values[:, -(self.sequence_length)::,:]
        seq_src_key_padding_mask = self.src_key_padding_mask[:, -(self.sequence_length)::]
        seq_src_key_padding_mask = torch.concat((seq_src_key_padding_mask,torch.unsqueeze(torch.zeros((self.n_envs, ), device = self.out_device).to(torch.bool), 1)),1)
            
        seq_tgt_key_padding_mask = self.tgt_key_padding_mask[:, -(self.sequence_length)::]
            
        return {
            'src_obs': obs_dict,
            'tgt_action_mu': sequence_action_mu,
            'tgt_action_std': sequence_action_std,
            'tgt_value': sequence_value,
            'src_padding_mask': seq_src_key_padding_mask,
            'tgt_padding_mask': seq_tgt_key_padding_mask
        }
    
    
    def get_reversed_order(self, tensor_names: Optional[List[str]] = None  ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Return every tensor in the reversed order
        -> the most recent timestamp will be at 0
           and the largest timestep will be at horizon_length
           
         
        Returns:
            Dict[str, torch.Tensor]: keys() => {'action', ''}
        """
        if tensor_names is None:
            tensor_names = self.tensor_names
            
        res_dict = {}
        for tensor_name in tensor_names:
            
            curr_tensor = getattr(self, tensor_name) # could be the obs dict
            
            if isinstance(curr_tensor, Dict):
                res_dict[tensor_name] = {obs_name: curr_tensor[obs_name][:, -self.horizon_length::] for obs_name in curr_tensor.keys()}
            else:
                res_dict[tensor_name] = torch.flip( curr_tensor[:, -self.horizon_length::], (1,))
            
        if len(tensor_names) == 1:
            return res_dict[tensor_name]
        return res_dict

            
            
    def calc_advantages(self, 
                        final_dones: torch.Tensor, 
                        final_pred_values: torch.Tensor,
                        gamma: float,
                        gae_lambda: float
                        ):
        """Discout the values and calculate the advantages for the horizon_len and add to the buffer

        Args:
            final_dones (torch.Tensor): shape (num_envs)
            final_pred_values (torch.Tensor): shape(num_envs, num_values) 
            gamma (float): discount factor
            gae_lambda (float): bootstrapping tradeoff
        """
        assert (self.buffer_length -1) - self.right_advantage_pointer == self.horizon_length, "The complete horizon length must be played before the advantage will be calculated"
        
        last_gae_lam = 0
        
        for i in range(self.horizon_length + 1):
            t = self.right_advantage_pointer + i # index in the uncalculated territory
            if i == 0:
                next_non_terminal = 1.0 - final_dones
                next_values = final_pred_values
            else:
                next_non_terminal = 1.0 - self.dones[:, t-1]
                next_values = self.values[:, t-1]
            next_non_terminal = next_non_terminal.unsqueeze(1)
                         
            # discounted differenct between last predicted values and current predicted values + reward
            delta = self.rewards[:, t] + gamma * next_values * next_non_terminal - self.values[:, t]    
            self.advantages[:, t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        
        # calculate returns 
        self.returns[:, self.right_advantage_pointer::] = self.advantages[:, self.right_advantage_pointer::] + self.values[:, self.right_advantage_pointer::]
        
        self.right_advantage_pointer = self.buffer_length -1
        
         
    def block_write(self):
        self.can_write = False
        
    def allow_write(self):
        self.can_write = True
        
        
    def __len__(self):
        return self.buffer_length
     
    def get_dataset(self, minibatch_size: int):
        return SequenceDataset(self, minibatch_size)
        
            
class SequenceDataset(Dataset):
    
    def __init__(self, 
                    buffer: SequenceBuffer,
                    minibatch_size: int,
                    runnign_mean_value: Optional[RunningMeanStd] = None
                    ) -> None:
             
        
        self.minibatch_size = minibatch_size    
        self.horizon_length = buffer.horizon_length
        self.sequence_length = buffer.sequence_length
        
        self.length = buffer.n_envs * buffer.horizon_length // self.minibatch_size 
        
        self.data_buffer = {}
        buffer_res_dict : Dict[str, Union[Dict, torch.Tensor]] = buffer.get_reversed_order()
        
        
        
        for key in buffer_res_dict.keys():
            
            if isinstance(buffer_res_dict[key], Dict):
                # obs dict
                self.data_buffer[key] = {obs_key : self.expand_and_compacify_tensor(buffer_res_dict[key][obs_key], self.sequence_length + 1 ) for obs_key in buffer_res_dict[key].keys()}
            else:
                # just tensor
                self.data_buffer[key] = self.expand_and_compacify_tensor(buffer_res_dict[key], self.sequence_length)
        
        
        self.normalize_values(runnign_mean_value)
                
    def normalize_values(self, running_mean_value: Optional[RunningMeanStd]):
        # TODO add this 
        
        advantages = torch.sum(self.data_buffer['advantages'], axis = 1)
        
        if self.normalize_values:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        pass
        
                
                
    def expand_and_compacify_tensor(self, tensor: torch.Tensor, sequence_length: int ) -> torch.Tensor:
        """Expand the shape of a tensor of shape (num_envs, buffer_length, c, ...)
        to the shape (num_envs, buffer_length, buffer_length, c, ...)
        cropped to (num_envs, horizon_length, sequence_length, c, ...)
        and than compactify it into shape ( num_envs * horizon_length, sequence_length, c, ...)

        Args:
            tensor (torch.Tensor): Tensor of shape (num_envs, buffer_length, c, ...)

        Returns:
            torch.Tensor: Tensor of shape ( num_envs * horizon_length, sequence_length, c, ...)
        """
        
        # create the new tensor shape (-1, -1, b, -1 ***)
        expanded_shape = (-1,-1,)+ (tensor.shape[1],) + ((-1,) * (len(tensor.shape) - 2))  
        expanded_tensor = tensor.unsqueeze(2).expand(expanded_shape)
        
        # crop the buffer_length to only be horizon_length and sequence_length
        expanded_tensor = expanded_tensor[:, 0:self.horizon_length + 1, 0:sequence_length, ]
        
        # contract the first two dimensions
        compacted_shape = (expanded_tensor.shape[0] * expanded_tensor.shape[1], expanded_tensor.shape[2] ,) + tensor.shape[2:]
        
        # reflip the sequence dimension
        expanded_tensor =  expanded_tensor.contiguous().view(compacted_shape) 
        
        # in the sequence, the oldest timestamp is at index 0 and the most recent at index sequence_length
        return torch.flip(expanded_tensor, [1]).contiguous() 
        
        
        
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """get the minibatch at the specified indes 

        Args:
            index (int): index ranging from 0 to len(self) -1

        Returns:
            Dict[str, torch.Tensor]: torch(minibatch_size, sequence_length, ) + singluar_tensor_shape
                                     The sequence_length dimension is ordered, so that the oldest timestamp is at the 0 position
        """
        
        start = index * self.minibatch_size
        end = start + self.minibatch_size
        
        out_dict = {}
        
        for key, buffer_item in self.data_buffer.items():
            if isinstance(buffer_item, Dict):
                # the network exp
                out_dict[key] = {obs_key: buffer_item[obs_key][start: end] for  obs_key in buffer_item.keys()}
            else:
                out_dict[key] = buffer_item[start: end]
                
        
        return out_dict
        
        
    def __len__(self):
        return self.length