import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class EncoderNet(nn.Module):
    def __init__(self, config_params=None):
        super(EncoderNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
