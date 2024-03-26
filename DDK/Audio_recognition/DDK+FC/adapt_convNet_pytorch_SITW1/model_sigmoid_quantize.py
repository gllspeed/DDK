import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from module import *
class DDK(nn.Module):
    def __init__(self):
        super(DDK, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        ############DDK###########################
        #nn.init.uniform_(self.alpha, 0.29, 0.4)
        #nn.init.constant_(self.alpha, 0.143179)
        #nn.init.uniform_(self.beta, 0.7, 0.9)
        #nn.init.constant_(self.beta,0.837902)
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



    def forward(self, x, target):

        x = self.DDK(x)

        ##############DDK#################
        x = F.sigmoid(self.fc1(x))
        x = self.fc(x)

        #x = self.fc1(x)
        #x = self.fc(x)

        return x, target

    def quantize(self, num_bits=8):
        self.qfc1 = QLinear(self.fc1, qi=True, qo=True, num_bits=num_bits)
        self.qsigmoid = QSigmoid_quantize(qi=True, qo=False, num_bits=num_bits)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)
    ############统计每一层的min, max, zero, scale
    def quantize_forward(self, x):
        x = self.DDK(x)
        x = self.qfc1(x)
        x = self.qsigmoid(x)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qfc1.freeze()
        self.qsigmoid.freeze()
        self.qfc.freeze(self.qfc1.qo)

    def quantize_inference(self, x, target):
        x = x.float()
        x = self.DDK(x)
        qx = self.qfc1.qi.quantize_tensor(x)
        '''
        input_uint8 = qx
        target = target.cpu().detach().numpy().flatten()[0]
        input = np.insert(input_uint8.cpu().detach().numpy().flatten(), 0, target)
        np.savetxt('/home/gaolili/deepLearning_project/pytorch-quantization-demo_xinpian/save_weights/inputs/'+'input_'+str(i)+'.txt', input, fmt='%d')
        '''
        qx = self.qfc1.quantize_inference(qx)
        x = self.qfc1.qo.dequantize_tensor(qx)
        qx = self.qsigmoid.quantize_inference(x)
        qx = self.qfc.quantize_inference(qx)

        out = self.qfc.qo.dequantize_tensor(qx)
        return out, target


