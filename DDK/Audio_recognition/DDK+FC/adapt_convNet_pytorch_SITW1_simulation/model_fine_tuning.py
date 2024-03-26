import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
class DDK(nn.Module):
    def __init__(self):
        super(DDK, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1), requires_grad = False)
        self.beta = nn.Parameter(torch.empty(1), requires_grad = False)
        ############DDK###########################
        #nn.init.uniform_(self.alpha, 0.29, 0.4)
        #nn.init.constant_(self.alpha, 0.354447)
        #nn.init.uniform_(self.beta, 0.7, 0.9)
        #nn.init.constant_(self.beta,0.854424)
        #nn.init.uniform_(self.alpha, 0.01, 0.1)
        #nn.init.uniform_(self.beta, 0.2, 0.3)
        nn.init.uniform_(self.alpha, 0.001, 1)
        nn.init.uniform_(self.beta, 0.001, 1)
        

    def forward(self, input):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        m, n = input.shape
        weight_matrix = torch.ones(m, n).to(device)

        for k in range(1, n):
            temp = 1 / weight_matrix[:, k - 1]
            weight_matrix[:, k] = weight_matrix[:, k - 1] + self.alpha * temp - self.beta * input[:, k - 1]


        return weight_matrix


class Net(nn.Module):
    def __init__(self, num_class=4, num_every_class=312):
        super(Net, self).__init__()
        self.num_class = num_class
        self.num_every_class = num_every_class

        self.DDK = DDK()
        ###########DDK#############################
        self.fc1 = nn.Linear(128, 16, bias=True)
        self.fc = nn.Linear(16, self.num_class, bias=True)



    def forward(self, x):

        x = self.DDK(x)

        ##############DDK#################
        x = torch.sigmoid(self.fc1(x))
        x = self.fc(x)

        #x = self.fc1(x)
        #x = self.fc(x)

        return x



