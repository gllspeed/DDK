import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from torch.autograd import Variable
from constants import  BATCH_SIZE_TRAIN, num_frames
from utils import DDK

logger = logging.getLogger(__name__)

class FC_BN(nn.Module):
    def __init__(self, fc_module, bn_module):
        super(FC_BN, self).__init__()
        self.fc_module = fc_module
        self.bn_module = bn_module

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.fc_module.weight * gamma_.view(self.fc_module.out_features, 1)
            if self.fc_module.bias is not None:
                bias = gamma_ * self.fc_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.fc_module.weight * gamma_
            if self.fc_module.bias is not None:
                bias = gamma_ * self.fc_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        #self.weight = nn.Parameter(weight)
        #self.bias = nn.Parameter(bias)
        return weight, bias




    def forward(self, x):


        if self.training:
            y = F.linear(x,
            self.fc_module.weight,
            self.fc_module.bias)

            y = y.contiguous().view(self.fc_module.out_features, -1) # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        x = F.linear(x, weight, bias)
        return x


class DeepSpeakerModel(nn.Module):
    def __init__(self, num_channels=1, include_softmax = True, num_speakers_softmax=10, num_data=BATCH_SIZE_TRAIN, is_DDK=True):
        super(DeepSpeakerModel, self).__init__()
        self.include_softmax = include_softmax
        self.num_speakers_softmax = num_speakers_softmax
        self.num_data = num_data
        self.is_DDK = is_DDK
        if self.is_DDK:
            self.beta = nn.Parameter(torch.empty(60))
            nn.init.uniform_(self.beta, 0.1, 1)
            self.alpha = nn.Parameter(torch.empty(60))
            nn.init.uniform_(self.alpha, 0.1, 1)


        ######################################

        self.fc1 = nn.Linear(4800, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, self.num_speakers_softmax, bias=False)

        #self.fc1_bn1 = FC_BN(self.fc1, self.bn1)
        #self.fc2_bn2 = FC_BN(self.fc2, self.bn2)
        #self.fc3_bn3 = FC_BN(self.fc3, self.bn3)

        ########################


    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

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
        fusedfc.bias.data = (torch.matmul(w_bn, b_fc) + b_bn)

        return fusedfc


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
        x = F.sigmoid(self.bn1(self.fc1(x)))
        x = F.sigmoid(self.bn2(self.fc2(x)))
        x = F.sigmoid(self.bn3(self.fc3(x)))

        x = self.fc4(x)
        if not self.training:
            sm = nn.Softmax(dim=1)
            x = sm(x)

        return x

