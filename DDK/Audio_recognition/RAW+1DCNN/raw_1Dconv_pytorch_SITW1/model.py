import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()

        ##############WhaleSong 4类数据集#################

        ##############Speaker 10类数据集#################
        self.conv1 = nn.Conv1d(1, 128, 3, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, 3, 1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(512, 512, bias=False)

        self.fc2 = nn.Linear(512, 10, bias=False)



    def forward(self, x):

        ##############Speaker 10类数据集#################
        x = x[:, np.newaxis, :]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 3)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 3)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 3)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 3)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool1d(x, 3)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = x.contiguous().view(-1, x.shape[1] * x.shape[2])
        x = self.fc1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


