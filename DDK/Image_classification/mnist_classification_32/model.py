import torch
import torch.nn as nn
import torch.nn.functional as F


class DDK(nn.Module):
    def __init__(self):
        super(DDK, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1))
        # nn.init.uniform_(self.alpha, 0.01, 0.1)
        # nn.init.uniform_(self.alpha, 0.29, 0.4)
        self.beta = nn.Parameter(torch.empty(1))
        # nn.init.uniform_(self.beta, 0.2, 0.3)
        # nn.init.uniform_(self.beta, 0.7, 0.85)
        nn.init.uniform_(self.alpha, 0.001, 1)
        nn.init.uniform_(self.beta, 0.001, 1)
        print("self.alpha", self.alpha)
        print("self.beta", self.beta)

    def forward(self, input):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #input = torch.transpose(input, 2, 3)
        l, p, m, n = input.shape
        weight_matrix = torch.ones(l ,p, m, n).to(device)

        for k in range(1, n):
            temp = 1 / weight_matrix[:, :, :, k - 1]
            weight_matrix[:, :, :, k] = weight_matrix[:, :, :,  k - 1] + self.alpha * temp - self.beta * input[:, :, :, k - 1]

        weight_matrix = torch.reshape(weight_matrix, (l, m*n))
        return weight_matrix



class Net(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()

        self.DDK = DDK()

        self.fc1 = nn.Linear(121, 32, bias=False)

        self.fc = nn.Linear(32, 10, bias=False)

        '''
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, groups=20)
        self.fc = nn.Linear(5*5*40, 10)
        '''
        '''
        ###############lenet-5##################
        self.conv1 = nn.Conv2d(num_channels, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16*5*5, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc = nn.Linear(84, 10, bias=True)
        '''
    def forward(self, x):

        x = self.DDK(x)
        x = F.sigmoid(self.fc1(x))

        x = self.fc(x)

        '''
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        '''
        '''
        ############lenet-5############
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        '''
        return x





