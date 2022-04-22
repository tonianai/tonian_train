
import torch.nn as nn
from typing import List, Optional
from tonian.training2.common.configuration_types import ActivationFn, MlpConfiguration, CnnConfiguration

class BaseNetwork(nn.Module):
    
    
    def __init__(self) -> None:
        super().__init__()
        
    def is_separate_critic(self):
            return False

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None
    
    
class A2CNetwork(BaseNetwork):
    
    def __init__(self) -> None:
        super().__init__()
        
        
class SeperatedA2CNetwork(A2CNetwork):
    
    def __init__(self, 
                 actor_cnn: Optional[CnnConfiguration],
                 critic_cnn: Optional[CnnConfiguration],
                 actor_units: MlpConfiguration,
                 critic_units: MlpConfiguration,
                 
                 ) -> None:
        super().__init__(activation_fn)
    
        

    