import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class EncoderNet(nn.Module):
    '''
    Encoder for the transcription model.
    '''
    def __init__(self, config_params=None):
        super(EncoderNet, self).__init__()
        if config_params:
            p = 1 - config_params['keep_prob']
        else: p = 0.005
        self.localization_network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(32*8*8, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.encoding_network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 8, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(8),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Linear(2048, 100)
        self._init_all_parameters()

    def _init_all_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        transformed_x = self.stn_forward(x)
        # transformed_x = x
        encoding = self.encoding_network(transformed_x)
        encoding = encoding.reshape(encoding.shape[0], -1)
        encoding = self.fc(encoding)
        return encoding, transformed_x

    def stn_forward(self, x):
        theta_prime = self.localization_network(x)
        theta_prime = self.regressor(
            theta_prime.reshape(theta_prime.shape[0], -1)
        )
        N, C = theta_prime.shape
        assert C == 2*3
        theta = theta_prime.reshape(N, 2, 3)
        mesh_grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, mesh_grid)
        return x
