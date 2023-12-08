from typing import Dict,  Optional
from tonian_train.networks.network_elements import *
from tonian_train.networks.sequential.base_seq_nn import BaseSequentialNet
from tonian_train.networks.simple_networks import A2CSimpleNet


class SequentialNetWrapper(BaseSequentialNet):
    
    def __init__(self) -> None:
        super().__init__()
        
        
    def __init__(self, 
                 simple_net: A2CSimpleNet) -> None:
        """_summary_
                                                        
                                                        
               Observations:                              Actions & Values:                                         
                                                        
               [...,obs(t-2),obs(t=1),obs(t=0)]           [...,action_mean(t-1), action_mean(t=0]
                                                          [...,action_std(t-1), action_std(t=0]
                                                          [...,value(t-1), value(t=0] 
                              |                            
                             \|/
                             
                         [obs(t=0)]                            
                              
                               |                              
                               |                                    
                              \|/             
                                              
                             |----------------------------------------|
                             |            A2C simple net              |
                             |----------------------------------------|
                              
                              
                            
                                    |                        |
                                    |                        |
                                   \|/                      \|/
                                    
                                Actions (mu & std)        States
                                    
 
        """
        super().__init__()
        self.simple_net = simple_net
     
    def forward(self, src_obs: Dict[str, torch.Tensor], 
                      tgt_action_mu: torch.Tensor,  
                      tgt_action_std: torch.Tensor,  
                      tgt_value: torch.Tensor, 
                      tgt_mask=None,
                      src_pad_mask=None,
                      tgt_pad_mask=None):
        """_summary_

        Args:
            src_obs (Dict[str, torch.Tensor]): tensor shape (batch_size, src_seq_length, ) + obs_shape  The current observations
            tgt_action_mu (torch.Tensor): shape (batch_size, tgt_seq_length, action_length) The actions made from the last observations
            tgt_action_std (torch.Tensor):  (batch_size, tgt_seq_length, action_length)
            tgt_value (torch.Tensor): (batch_size, sr)
            tgt_mask (_type_, optional): _description_. Defaults to None.
            src_pad_mask (_type_, optional): _description_. Defaults to None.
            tgt_pad_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        obs = {}
        # cut of the rest in favor of only the last obsevation
        for obs_key in src_obs.keys():
            obs[obs_key] = src_obs[obs_key][:, -1].squeeze() 

        return self.simple_net(obs)