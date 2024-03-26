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
        self.fc1 = nn.Linear(784, 72, bias=True)
        self.bn1 = nn.BatchNorm1d(72)
        self.fc = nn.Linear(72, 10, bias=True)


    def forward(self, x):

        x = self.DDK(x)
        #x = F.sigmoid(self.fc1(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc(x)


        return x





