import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from training_utils import *
import tracemalloc
import gc


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, leak=0.5, gamma=1.0, soft_reset=False):
        """
        Implementing the LIF neurons.
        @param thresh: firing threshold;
        @param tau: membrane potential decay factor;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param soft_reset: whether using soft-reset or hard-reset.
        """
        super(LIFSpike, self).__init__()

        self.thresh = thresh
        self.leak = leak
        self.gamma = gamma
        self.soft_reset = soft_reset

        self.membrane_potential = 0

    def reset_mem(self):
        self.membrane_potential = 0

    def forward(self, s):

        H = s + self.membrane_potential
        
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()

        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)

        return s
    