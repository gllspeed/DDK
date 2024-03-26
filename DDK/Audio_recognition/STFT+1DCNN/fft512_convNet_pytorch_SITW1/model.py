import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()


        ##############Speaker 10类数据集#################
        self.conv1 = nn.Conv1d(1, 32, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, 3, 1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, padding=1, bias=False)
        self.fc = nn.Linear(32 * 64, 10, bias=False)


    def forward(self, x):

        ##############Speaker 10类数据集#################
        x = x[:, np.newaxis, :]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        #x = F.relu(self.conv3(x))
        #x = F.max_pool1d(x, 2)
        #print("x.shape", x.shape)
        # x = x.view(-1, 5 * 5 * 40)
        x = x.view(-1, 64 * 32)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x


