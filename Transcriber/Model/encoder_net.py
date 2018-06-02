import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

class EncoderNet(nn.Module):
    '''
    Encoder for the transcription model.
    '''
    def __init__(
        self, config_params=None,
        yolo_weights_path='../Data/FullData/yolo.h5'
    ):
        super(EncoderNet, self).__init__()
        if config_params:
            p = 1 - config_params['keep_prob']
        else: p = 0.005
        self.p = p
        self.yolo_weights = h5py.File(yolo_weights_path, 'r')

        '''
        self.localization_network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(16*16*16, 3*2),
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        '''
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
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=p, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Linear(2*2*32, 100)

        self.auxilary = nn.Linear(100, 1)
        self._init_some_parameters()
        self._init_full_yolo()

    def _init_some_parameters(self):
        for m in self.encoding_network.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        transformed_x = x #self.stn_forward(x)
        encoding = self._run_block_two(self._run_block_one(transformed_x))
        encoding = self.encoding_network(encoding)
        encoding = encoding.reshape(encoding.shape[0], -1)
        encoding = self.fc(encoding)

        class_ = self.auxilary(encoding)
        return encoding, None, class_

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

    ################################################
    # In a feat of bery bad Engineering design     #
    # We copy code from the Localizer/Model/net.py #
    # This code initializes YOLO9000.              #
    #################################################
    def _init_full_yolo(self, K=7, B=2, S=3):
        # Block 1 #
        p = self.p
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1, stride=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout(p=p, inplace=True)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, stride=1,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout(p=p, inplace=True)

        # Block 2 #
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=1,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.d3 = nn.Dropout(p=p, inplace=True)
        self.conv4 = nn.Conv2d(
            128, 64, kernel_size=1, padding=0, stride=1,
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.d4 = nn.Dropout(p=p, inplace=True)
        self.conv5 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=1,
        )
        self.bn5 = nn.BatchNorm2d(128)
        self.d5 = nn.Dropout(p=p, inplace=True)
        self.mp5 = nn.MaxPool2d(2, 2)

        # Block 3 #
        self.conv6 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, stride=1
        )
        self.bn6 = nn.BatchNorm2d(256)
        self.d6 = nn.Dropout(p=p, inplace=True)
        self.conv7 = nn.Conv2d(
            256, 128, kernel_size=1, padding=0, stride=1
        )
        self.bn7 = nn.BatchNorm2d(128)
        self.d7 = nn.Dropout(p=p, inplace=True)
        self.conv8 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, stride=1
        )
        self.bn8 = nn.BatchNorm2d(256)
        self.d8 = nn.Dropout(p=p, inplace=True)
        self.mp8 = nn.MaxPool2d(2, 2)

        # Block 4 #
        self.conv9 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, stride=1
        )
        self.bn9 = nn.BatchNorm2d(512)
        self.d9 = nn.Dropout(p=p, inplace=True)
        self.conv10 = nn.Conv2d(
            512, 256, kernel_size=1, padding=0, stride=1
        )
        self.bn10 = nn.BatchNorm2d(256)
        self.d10 = nn.Dropout(p=p, inplace=True)

        # Output Layer. #
        self.conv14 = nn.Conv2d(
            256, 3, kernel_size=3, padding=1, stride=1
        )
        self.mp14 = nn.MaxPool2d(2, 2)
        init.xavier_uniform_(self.conv14.weight.data)
        self._init_full_yolo_weights()

    def _run_block_one(self, x):
        out = F.leaky_relu(self.d1(self.bn1(self.conv1(x))))
        out = F.leaky_relu(self.d2(self.bn2(self.conv2(out))))
        out = F.leaky_relu(self.d3(self.bn3(self.conv3(out))))
        out = F.leaky_relu(self.d4(self.bn4(self.conv4(out))))
        out = self.mp5(F.leaky_relu(self.d5(self.bn5(self.conv5(out)))))
        return out

    def _run_block_two(self, x):
        out = F.leaky_relu(self.d6(self.bn6(self.conv6(x))))
        out = F.leaky_relu(self.d7(self.bn7(self.conv7(out))))
        out = self.mp8(F.leaky_relu(self.d8(self.bn8(self.conv8(out)))))
        out = F.leaky_relu(self.d9(self.bn9(self.conv9(out))))
        out = F.leaky_relu(self.d10(self.bn10(self.conv10(out))))
        return self.mp14(self.conv14(out))

    def _init_full_yolo_weights(self):
        for i in range(10):
            requires_grad = (i+1 >= 1)
            conv_name = 'conv2d_' + str(i+1)
            bn_name = 'batch_normalization_' + str(i+1)
            cw = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][conv_name][conv_name][u'kernel:0']))
            )).permute(3, 2, 0, 1)
            bn_beta = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][bn_name][bn_name][u'beta:0']))
            ))
            bn_gamma = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][bn_name][bn_name][u'gamma:0']))
            ))
            bn_mean = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][bn_name][bn_name][u'moving_mean:0']))
            ))
            bn_var = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][bn_name][bn_name][u'moving_variance:0']))
            ))

            conv_expr = 'self.conv' + str(i+1); bn_expr = 'self.bn' + str(i+1)
            eval(conv_expr).weight = nn.Parameter(cw, requires_grad=requires_grad)
            eval(bn_expr).running_mean = nn.Parameter(bn_mean, requires_grad=False)
            eval(bn_expr).running_var = nn.Parameter(bn_var, requires_grad=False)
            eval(bn_expr).weight = nn.Parameter(bn_gamma, requires_grad=requires_grad)
            eval(bn_expr).bias = nn.Parameter(bn_beta, requires_grad=requires_grad)
        self.yolo_weights.close()
