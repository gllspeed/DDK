import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from constants import  BATCH_SIZE

logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, num_channels=1, include_softmax = True, num_class=10, num_data=BATCH_SIZE):
        super(Model, self).__init__()
        self.include_softmax = include_softmax
        self.num_class = num_class
        self.num_data = num_data

        #############C3D#########################
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, self.num_class)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        '''
        ###########3dConv###############
        self.conv1 = nn.Conv3d(1, 32, (3, 3, 3), 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 32, (3, 3, 3), 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(32)

        self.fc1 = nn.Linear(20 * 10 * 7 * 32, 256, bias=False)
        self.fc2 = nn.Linear(256, self.num_class, bias=False)
        '''
        ######################3*3



        ##############全连接########################



    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x

