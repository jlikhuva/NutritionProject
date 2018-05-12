import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LocalizerNet(nn.Module):
    def __init__(self, yolo_weights_path, config_params=None, use_full_yolo=False):
        super(LocalizerNet, self).__init__()
        self.yolo_weights = h5py.File(yolo_weights_path, 'r')
        if config_params:
            self.p = 1 - config_params['keep_prob']
        else:
            self.p = 0.0
        self.use_full_yolo = use_full_yolo
        if use_full_yolo:
            self._init_full_yolo()
        else:
            self._init_custom_model()

    def forward(self, x):
        if self.use_full_yolo:
            return self._full_yolo_forward(x)
        else:
            return self._custom_model_forward(x)

    def _init_custom_model(self):
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
        p = self.p
        self.bottle_neck_conv4 = nn.Conv2d(
            128, 32, kernel_size=1,
            padding=0, stride=1
        )
        self.bottle_neck_batchnorm4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout(p=p)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            32, 64, kernel_size=3,
            padding=1, stride=1
        )
        self.conv5_batchnorm = nn.BatchNorm2d(64)
        self.dropout5 = nn.Dropout(p=p)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(
            64, 128, kernel_size=3,
            padding=1, stride=1
        )
        self.conv6_batchnorm = nn.BatchNorm2d(128)
        self.dropout6 = nn.Dropout(p=p)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv7 = nn.Conv2d(
        #     256, 512, kernel_size=3,
        #     padding=1, stride=1
        # )
        # self.conv7_batchnorm = nn.BatchNorm2d(512)
        # self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottle_neck_conv8 = nn.Conv2d(
            128, 8, kernel_size=1,
            padding=0, stride=1
        )
        self.bottle_neck_batchnorm8 = nn.BatchNorm2d(8)
        self.dropout8 = nn.Dropout(p=p)

        self.fc = nn.Linear(15840, 550)

    def _custom_model_forward(self, x):
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
            self.dropout4(
                self.bottle_neck_batchnorm4(
                    self.bottle_neck_conv4(out)
        )))
        out = self.pool4(out)

        out = F.relu(
            self.dropout5(
                self.conv5_batchnorm(
                    self.conv5(out)
        )))
        out = self.pool5(out)

        out = F.relu(
            self.dropout6(
                self.conv6_batchnorm(
                    self.conv6(out)
        )))
        out = self.pool6(out)

        # out = F.relu(
        #     self.conv7_batchnorm(
        #         self.conv7(out)
        # ))
        # out = self.pool7(out)

        out = F.relu(
            self.dropout8(
                self.bottle_neck_batchnorm8(
                    self.bottle_neck_conv8(out)
        )))

        #######
        # FC  #
        ######
        out = self.fc(out.view(out.shape[0], -1))
        return out

    def _init_full_yolo(self):
        # Block 1 #
        p = self.p
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, padding=1, stride=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout(p=p)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, stride=1,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout(p=p)
        self.mp2 = nn.MaxPool2d(2, 2)

        # Block 2 #
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=1,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.d3 = nn.Dropout(p=p)
        self.conv4 = nn.Conv2d(
            128, 64, kernel_size=1, padding=0, stride=1,
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.d4 = nn.Dropout(p=p)
        self.conv5 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, stride=1,
        )
        self.bn5 = nn.BatchNorm2d(128)
        self.d5 = nn.Dropout(p=p)
        self.mp5 = nn.MaxPool2d(2, 2)

        # Block 3 #
        self.conv6 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, stride=1
        )
        self.bn6 = nn.BatchNorm2d(256)
        self.d6 = nn.Dropout(p=p)
        self.conv7 = nn.Conv2d(
            256, 128, kernel_size=1, padding=0, stride=1
        )
        self.bn7 = nn.BatchNorm2d(128)
        self.d7 = nn.Dropout(p=p)
        self.conv8 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, stride=1
        )
        self.bn8 = nn.BatchNorm2d(256)
        self.d8 = nn.Dropout(p=p)
        self.mp8 = nn.MaxPool2d(2, 2)

        # Block 4 #
        self.conv9 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, stride=1
        )
        self.bn9 = nn.BatchNorm2d(512)
        self.d9 = nn.Dropout(p=p)
        self.conv10 = nn.Conv2d(
            512, 256, kernel_size=1, padding=0, stride=1
        )
        self.bn10 = nn.BatchNorm2d(256)
        self.d10 = nn.Dropout(p=p)
        self.conv11 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, stride=1
        )
        self.bn11 = nn.BatchNorm2d(512)
        self.d11 = nn.Dropout(p=p)
        self.conv12 = nn.Conv2d(
            512, 256, kernel_size=1, padding=0, stride=1
        )
        self.bn12 = nn.BatchNorm2d(256)
        self.d12 = nn.Dropout(p=p)
        self.conv13 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1, stride=1
        )
        self.bn13 = nn.BatchNorm2d(512)
        self.d13 = nn.Dropout(p=p)
        self.mp13 = nn.MaxPool2d(2, 2)

        # Output Layer. #
        self.conv14 = nn.Conv2d(
            512, 550, kernel_size=(15, 8), padding=0, stride=1
        )
        self._init_full_yolo_weights()


    def _full_yolo_forward(self, x):
        out = self._run_block_one(x)
        out = self._run_block_two(out)
        return out.reshape(out.shape[0], out.shape[1])

    def _run_block_one(self, x):
        out = self.mp1(F.leaky_relu(self.d1(self.bn1(self.conv1(x)))))
        out = self.mp2(F.leaky_relu(self.d2(self.bn2(self.conv2(out)))))
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
        out = F.leaky_relu(self.d11(self.bn11(self.conv11(out))))
        out = F.leaky_relu(self.d12(self.bn12(self.conv12(out))))
        out = self.mp13(F.leaky_relu(self.d13(self.bn13(self.conv13(out)))))
        return self.conv14(out)

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

    def _init_full_yolo_weights(self):
        for i in range(13):
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
            eval(conv_expr).weight = nn.Parameter(cw, requires_grad=True)
            eval(bn_expr).running_mean = nn.Parameter(bn_mean, requires_grad=False)
            eval(bn_expr).running_var = nn.Parameter(bn_var, requires_grad=False)
            eval(bn_expr).weight = nn.Parameter(bn_gamma, requires_grad=True)
            eval(bn_expr).bias = nn.Parameter(bn_beta, requires_grad=True)
        self.yolo_weights.close()
