import torch
import torch.nn as nn
import torch.nn.functional as F

#from module_tf import *
from module_lsq import LinearLSQ, InputLSQ, SigmoidLSQ, SigmoidLSQ_quantize
import numpy as np

class MFL(nn.Module):
    def __init__(self, alpha=None, beta=None):
        super(MFL, self).__init__()
        if alpha != None and beta != None:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        else:
            self.alpha = nn.Parameter(torch.empty(1))
            self.beta = nn.Parameter(torch.empty(1))

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

        self.MFL = MFL()
        ###########MFL#############################
        self.fc1 = nn.Linear(128, 16, bias=True)
        self.fc = nn.Linear(16, self.num_class, bias=True)

    def forward(self, x):
        x = self.MFL(x)

        ##############MFL#################
        x = F.sigmoid(self.fc1(x))
        x = self.fc(x)

        # x = self.fc1(x)
        # x = self.fc(x)

        return x


class Net_quantize(nn.Module):
    def __init__(self, alpha=None, beta=None, Net_parameters=None, num_channels=1, num_bits=8):
        super(Net_quantize, self).__init__()
        if alpha and beta:
            self.MFL = MFL(alpha, beta)
        else:
            self.MFL = MFL()
        self.qAct1 = InputLSQ(nbits_a=num_bits)
        self.qAct2 = SigmoidLSQ_quantize(nbits_a=num_bits)
        self.qAct3 = InputLSQ(nbits_a=num_bits)
        if Net_parameters:
            self.qfc1 = LinearLSQ(128, 16, bias=True, nbits_w=num_bits, model_float=Net_parameters[0])
            self.qfc = LinearLSQ(16, 10, bias=True, nbits_w=num_bits, model_float=Net_parameters[1])
        else:
            self.qfc1 = LinearLSQ(128, 16, bias=True, nbits_w=num_bits)
            self.qfc = LinearLSQ(16, 10, bias=True, nbits_w=num_bits)




    def forward(self, x):
        x = self.MFL(x)
        x = self.qAct1(x)
        x = self.qfc1(x, qi_alpha = self.qAct1.alpha.data)
        x = self.qAct2(x)
        x = self.qAct3(x)
        x = self.qfc(x, qi_alpha = self.qAct3.alpha.data)
        return x


    def quantize_inference(self, x, alpha_dict):
        x = self.MFL(x)
        qx, _ = self.qAct1.quantize_inference(x)
        qx = self.qfc1.quantize_inference(qx, qi_alpha = alpha_dict['qAct1.alpha'], qo_alpha = alpha_dict['qAct2.alpha'])
        qx = self.qAct2.quantize_inference(qx)
        out = self.qfc.quantize_inference(qx, qi_alpha = alpha_dict['qAct3.alpha'])

        return out

    def add_loss(self, model, l2_alpha):
        alpha_loss = []
        for name, parameters in model.named_parameters():
            if 'alpha' in name:
                parameter = parameters.cpu().detach().numpy()
                parameter_target = np.sign(parameter) * 2 ** (np.round(np.log2(np.abs(parameter))))
                alpha_loss.append(torch.tensor(np.mean(abs(parameter - parameter_target))))
                #alpha_loss.append(torch.tensor(np.mean(np.square(parameter - parameter_target))))
        return l2_alpha * sum(alpha_loss)

