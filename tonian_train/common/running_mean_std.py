import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Tuple

'''
updates statistic from a full data
'''
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, is_sequence:bool = False,  norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only 
        
        if is_sequence:
            self.axis = [0,1]
        else:
            self.axis = [0]
            
        in_size = insize
        

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

        current_mean = self.running_mean  
        current_var = self.running_var
        # get output


        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

class RunningMeanStdObs(nn.Module):
    def __init__(self, insize: Dict[str, Tuple], is_sequence:bool = False,  epsilon=1e-05, norm_only=False):
        
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon,is_sequence,  norm_only) for k,v in insize.items()
        })
    
    def forward(self, input, unnorm=False):

        res = {k : self.running_mean_std[k](v, unnorm) for k,v in input.items()}
        return res