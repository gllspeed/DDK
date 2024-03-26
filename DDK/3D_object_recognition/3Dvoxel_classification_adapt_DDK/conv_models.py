import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from constants import  BATCH_SIZE_TRAIN, BATCH_SIZE_TEST
from utils import DDK

logger = logging.getLogger(__name__)

class DeepSpeakerModel(nn.Module):
    def __init__(self, num_channels=1, include_softmax = True, num_speakers_softmax=10, num_data=BATCH_SIZE_TRAIN, is_DDK=True):
        super(DeepSpeakerModel, self).__init__()
        self.include_softmax = include_softmax
        self.num_speakers_softmax = num_speakers_softmax
        self.num_data = num_data
        self.is_DDK = is_DDK
        #self.alpha_init = alpha_init
        #self.beta_init = beta_init
        if self.is_DDK:
            '''
            self.beta = nn.Parameter(self.beta_init)
            self.alpha = nn.Parameter(self.alpha_init)
            '''

            #self.beta = nn.Parameter(torch.empty(32))
            self.beta = nn.Parameter(torch.empty(32))
            #nn.init.constant_(self.beta, 0.21)
            nn.init.uniform_(self.beta, 0.1, 1)
            #self.alpha = nn.Parameter(torch.empty(32))
            self.alpha = nn.Parameter(torch.empty(32))
            #nn.init.constant_(self.alpha, 0.56)
            nn.init.uniform_(self.alpha, 0.1, 1)
        '''
        self.fc1 = nn.Linear(1024, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128,  bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, self.num_speakers_softmax, bias=False)
        '''
        self.fc1 = nn.Linear(1024, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, self.num_speakers_softmax, bias=False)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
    def forward(self, x):

        ##############5*5#################
        if self.is_DDK:
            x = DDK(x, self.alpha, self.beta)
            ###########查看DDK之后数据的分布##################
            '''
            import matplotlib.pyplot as plt
            y = x[:, 0].detach().cpu().numpy()
            x = [i for i in range(25600)]
            plt.plot(x, y)
            plt.show()
            '''

            #####################

            x = F.sigmoid(self.bn1(self.fc1(x)))
            x = F.sigmoid(self.bn2(self.fc2(x)))
            x = F.sigmoid(self.bn3(self.fc3(x)))
            #x = F.dropout(x, p = 0.7, training=self.training)
            x = self.fc4(x)
            if not self.training:
                sm = nn.Softmax(dim=1)
                x = sm(x)


        '''
        x = F.relu(self.fc1(x))

        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        sm = nn.Softmax(dim=1)
        x = sm(x)
        '''
        return x

