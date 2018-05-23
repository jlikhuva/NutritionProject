import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DecoderNet(nn.Module):
    def __init__(
        self, hidden_size=512, output_size=622,
        dropout_p=0.0, max_length=120
    ):
        super(DecoderNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
