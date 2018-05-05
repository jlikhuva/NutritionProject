import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LocalizerNet(nn.Module):
    def __init__(self, yolo_weights_path):
        super(LocalizerNet, self).__init__()
        self.yolo_weights = h5py.File(yolo_weights_path, 'r')

        ########################
        #  Frozen Layers      #
        #######################
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1, stride=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, stride=1,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=1,
            bias=False
        )
        self._init_frozen_layers()


        #####################
        # Trainable Layers  #
        ####################
        self.bottle_neck_conv4 = nn.Conv2d(
            128, 64, kernel_size=1,
            padding=0, stride=1
        )
        self.bottle_neck_batchnorm4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            64, 128, kernel_size=3,
            padding=1, stride=1
        )
        self.conv5_batchnorm = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(
            128, 256, kernel_size=3,
            padding=1, stride=1
        )
        self.conv6_batchnorm = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(
            256, 512, kernel_size=3,
            padding=1, stride=1
        )
        self.conv7_batchnorm = nn.BatchNorm2d(512)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottle_neck_conv8 = nn.Conv2d(
            512, 32, kernel_size=1,
            padding=0, stride=1
        )
        self.bottle_neck_batchnorm8 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(63360, 550)


    def forward(self, x):
        ######################
        # Feature Extraction #
        #####################
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        #####################
        # Fine Tuning       #
        ####################
        out = F.relu(
            self.bottle_neck_batchnorm4(
                self.bottle_neck_conv4(out)
        ))
        out = self.pool4(out)

        out = F.relu(
            self.conv5_batchnorm(
                self.conv5(out)
        ))
        out = self.pool5(out)

        out = F.relu(
            self.conv6_batchnorm(
                self.conv6(out)
        ))
        out = self.pool6(out)

        out = F.relu(
            self.conv7_batchnorm(
                self.conv7(out)
        ))
        out = self.pool7(out)

        out = F.relu(
            self.bottle_neck_batchnorm8(
                self.bottle_neck_conv8(out)
        ))

        #######
        # FC  #
        ######
        out = self.fc(out.view(out.shape[0], -1))
        return out

    def _init_frozen_layers(self):
        '''
            read saved weights, convert to NCHW,
            convert to pytorch Tensor, use tensor
            to init frozen layers, set then to not
            be updated.
        '''
        modules = [self.conv1, self.conv2, self.conv3]
        for i in range(3):
            name = 'conv2d_' + str(i+1)
            w = torch.from_numpy(np.array(
                list((self.yolo_weights['model_weights'][name][name][u'kernel:0']))
            )).permute(3, 2, 0, 1)
            modules[i].weight = nn.Parameter(w, requires_grad=False)
        self.yolo_weights.close()
