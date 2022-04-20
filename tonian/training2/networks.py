
import torch.nn as nn
from typing import List, Optional
from tonian.training2.common.types import ActivationFn, SequentialLayerConfiguration, CnnConfiguration

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
    
    def __init__(self, activation_fn: ActivationFn) -> None:
        super().__init__()
        
        
class SeperatedA2CNetwork(A2CNetwork):
    
    def __init__(self, 
                 activation_fn: ActivationFn,
                 cnn: Optional[CnnConfiguration],
                 actor_units: SequentialLayerConfiguration,
                 critic_units: SequentialLayerConfiguration,
                 
                 ) -> None:
        super().__init__(activation_fn)
    
        

    