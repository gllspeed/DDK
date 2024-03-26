import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()

        ###############lenet-5##################
        self.conv1 = nn.Conv2d(num_channels, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16*5*5, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc = nn.Linear(84, 10, bias=True)

    def forward(self, x):

        ############lenet-5############
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)

        return x





