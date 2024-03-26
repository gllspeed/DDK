import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

class Net(nn.Module):
    def __init__(self, num_class=4, num_data=372):
        super(Net, self).__init__()
        self.num_class = num_class
        self.num_data = num_data
        ##############WhaleSong 4类数据集#################
        self.fc1 = nn.Linear(41, 32, bias=True)

        self.fc = nn.Linear(32, self.num_class, bias=True)


    def forward(self, x):
        ##############WhaleSong 4类数据集#################
        x = self.fc1(x)
        x = self.fc(x)


        return x





