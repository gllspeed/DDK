import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from constants import  BATCH_SIZE_TRAIN, num_frames
from utils import DDK

logger = logging.getLogger(__name__)

class DeepSpeakerModel(nn.Module):
    def __init__(self, num_channels=1, include_softmax = True, num_speakers_softmax=10, num_data=BATCH_SIZE_TRAIN, is_DDK=True):
        super(DeepSpeakerModel, self).__init__()
        self.include_softmax = include_softmax
        self.num_speakers_softmax = num_speakers_softmax
        self.num_data = num_data
        self.is_DDK = is_DDK
        if self.is_DDK:
            self.beta = nn.Parameter(torch.empty(1), requires_grad = False)
            nn.init.uniform_(self.beta, 0.1, 1)
            self.alpha = nn.Parameter(torch.empty(1), requires_grad = False)
            nn.init.uniform_(self.alpha, 0.1, 1)



        ########################

        self.fc1 = nn.Linear(4800, 128, bias=False)

        self.fc2 = nn.Linear(128, 512, bias=False)

        self.bn2 = nn.BatchNorm1d(512)


        self.fc3 = nn.Linear(512, self.num_speakers_softmax, bias=False)


    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def fuse_fc_and_bn(self, fc, bn):
        # 初始化
        fusedfc = nn.Linear(
            fc.in_features,
            fc.out_features,
            bias=True
        )
        w_conv = fc.weight.clone().view(fc.out_features, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        # 融合层的权重初始化(W_bn*w_conv(卷积的权重))
        fusedfc.weight.data = (torch.mm(w_bn, w_conv).view(fusedfc.weight.size()))

        if fc.bias is not None:
            b_fc = fc.bias
        else:
            b_fc = torch.zeros(fc.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        # 融合层偏差的设置
        if fc.bias is not None:
            fusedfc.bias.data = (torch.matmul(w_bn.clone(), b_fc) + b_bn)
        else:
            fusedfc.bias.data = b_bn

        return fusedfc

    def fuse(self):
        self.fuse_fc1_and_bn1 = self.fuse_fc_and_bn(self.fc1, self.bn1)
        self.fuse_fc2_and_bn2 = self.fuse_fc_and_bn(self.fc2, self.bn2)
        self.fuse_fc3_and_bn3 = self.fuse_fc_and_bn(self.fc3, self.bn3)
    def forward(self, x):

        ##############5*5#################
        if self.is_DDK:
            x = DDK(x, self.alpha, self.beta)
            ###########查看DDK之后数据的分布##################
            '''
            import matplotlib.pyplot as plt
            y1 = x[:, 0].detach().cpu().numpy()
            x1 = [i for i in range(6944)]

            y2 = x[:, 1].detach().cpu().numpy()
            y3 = x[:, 2].detach().cpu().numpy()
            plt.plot(x1, y1)
            plt.plot(x1, y2)
            plt.plot(x1, y3)
            plt.show()
            '''

        #####################

        if self.training:
            x = F.relu(self.fc1(x))

            x = F.relu(self.bn2(self.fc2(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc3(x)


        else:
            '''
            x = F.sigmoid(self.fuse_fc1_and_bn1(x))
            x = F.sigmoid(self.fuse_fc2_and_bn2(x))
            x = F.sigmoid(self.fuse_fc3_and_bn3(x))
            x = self.fc4(x)
            sm = nn.Softmax(dim=1)
            x = sm(x)
            '''

            ''''
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
            x = self.fc4(x)
            if not self.training:
                sm = nn.Softmax(dim=1)
                x = sm(x)
            '''


            '''
            x = self.fc1(x)
            mean = torch.mean(x,dim = 0)
            var = torch.std(x, dim = 0)

            #self.bn1.running_mean.data = mean
            #self.bn1.running_var.data = var
            x = self.bn1(x)
            x = F.sigmoid(x)

            x = self.fc2(x)
            mean = torch.mean(x, dim=0)
            var = torch.std(x, dim=0)
            self.bn2.running_mean.data = mean
            self.bn2.running_var.data = var
            x = self.bn2(x)
            x = F.sigmoid(x)

            x = self.fc3(x)
            mean = torch.mean(x, dim=0)
            var = torch.std(x, dim=0)
            self.bn3.running_mean.data = mean
            self.bn3.running_var.data = var
            x = self.bn3(x)
            x = F.sigmoid(x)

            x = self.fc4(x)
            sm = nn.Softmax(dim=1)
            x = sm(x)
            '''

            x = F.relu(self.fc1(x))

            x = F.relu(self.bn2(self.fc2(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc3(x)
            sm = nn.Softmax(dim=1)
            x = sm(x)


        return x

