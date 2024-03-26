
###########浮点转定点转整型的量化方案，scale正好是x的二次幂
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from function import FakeQuantize

def calcScalePointBit_tf(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = float((max_val - min_val) / (qmax - qmin))

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax

    zero_point = int(zero_point)

    return scale

def calcScalePointBit(min_val, max_val, num_bits=8):
    qmin = - 2. ** (num_bits - 1)
    qmax = 2. ** (num_bits - 1) - 1
    min_diff = float('inf')

    point_bit = 0
    ######定点最大值：(2**i - 2**(-(num_bits-1-i)))
    ######定点最小值：(-2**i)
    ######浮点最大值：max_val
    ######浮点最小值：min_val

    ############################浮点映射到定点，寻找合适的整数位数，小数位数###############
    for i in range(num_bits):
        diff = abs(((2**i - 2**(-(num_bits-1-i))) - (-2**i)) - (max_val - min_val))
        if diff < min_diff:
            min_diff = diff
            point_bit = i
    #################################

    
    scale = 2**(-(num_bits-1-point_bit))
    return scale

def calcScalePointBit_sigmoid(min_val, max_val, num_bits=8):
    qmin = - 2. ** (num_bits - 1)
    qmax = 2. ** (num_bits - 1) - 1
    scale = float((max_val - min_val) / (qmax - qmin))
    return scale


def quantize_tensor(x, scale, num_bits=8, signed=True):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1.

    q_x = x*1./scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x


def quantize_tensor_uint8(x, scale, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    q_x = x * 1. / scale
    q_x.clamp_(qmin, qmax).round_()

    return q_x


def quantize_tensor_raodong(x, scale, num_bits=8, signed=True):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    q_x = x * 1. / scale
    q_x.clamp_(qmin, qmax).round_()
    random_mask = np.random.randint(0, 2, tuple(q_x.shape))
    random_mask[random_mask==0]=-1
    random_numpy =(q_x*0.1).int().cpu() * random_mask
    q_x_raodong = q_x.cpu() + random_numpy
    q_x_raodong.clamp_(qmin, qmax).round_()
    return q_x_raodong.cuda()
 
def dequantize_tensor(q_x, scale, num_bits):
    return q_x*scale


def search(M):
    P = 7000
    n = 1
    while True:
        Mo = int(round(2 ** n * M))
        # Mo 
        approx_result = Mo * P >> n
        result = int(round(M * P))
        error = approx_result - result

        print("n=%d, Mo=%f, approx=%d, result=%d, error=%f" % \
            (n, Mo, approx_result, result, error))

        if math.fabs(error) < 1e-9 or n >= 22:
            return Mo, n
        n += 1


class QParam:

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None
    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        
        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        
        self.scale = calcScalePointBit(self.min, self.max, self.num_bits)

    def update_sigmoid(self, tensor):
        self.max = torch.tensor(8)
        self.min = torch.tensor(-8)
        self.scale = calcScalePointBit_sigmoid(self.min, self.max, self.num_bits)
        #self.scale = calcScalePointBit(self.min, self.max, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, num_bits=self.num_bits)

    def quantize_tensor_uint8(self, tensor, num_bits = 8):
        #return quantize_tensor_uint8(tensor, self.scale, num_bits=self.num_bits)
        return quantize_tensor_uint8(tensor, self.scale, num_bits=num_bits)
    #################添加权重量化后扰动##################
    def quantize_tensor_raodong(self, tensor):
        return quantize_tensor_raodong(tensor, self.scale, num_bits=self.num_bits)

    ####################
    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, num_bits=self.num_bits)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info



class QModule(nn.Module):

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale
        print(print("fc_bias", self.qw.scale * self.qi.scale))
        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)

        ###############添加权重扰动################
        #self.fc_module.weight.data = self.qw.quantize_tensor_raodong(self.fc_module.weight.data)


        #self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        if self.fc_module.bias is not None:
            self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                       num_bits=32, signed=True)

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        x = self.fc_module(x)
        x = self.M * x
        x.round_()
        x.clamp_(- 2. ** (self.num_bits - 1), 2. ** (self.num_bits - 1) - 1).round_()
        return x


class QSigmoid_noquantize(QModule):

    def __init__(self, qi=False, qo=False, num_bits=None):
        super(QSigmoid_noquantize, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

    def forward(self, x):
        x = F.sigmoid(x)
        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x
    
    def quantize_inference(self, x):
        x = F.sigmoid(x)
        return x


class QSigmoid_quantize(QModule):

    def __init__(self, qi=False, qo=False, num_bits=None):
        super(QSigmoid_quantize, self).__init__(qi=qi, qo=qo, num_bits=num_bits)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        x.clamp_(-8, 8)
        if hasattr(self, 'qi'):
            self.qi.update_sigmoid(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.sigmoid(x)
        return x


    def quantize_inference(self, x):
        x.clamp_(-8, 8)
        x = self.qi.quantize_tensor(x)
        return x


class QReLU(QModule):

    def __init__(self, qi=False, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.relu(x)

        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < 0.] = 0.
        return x

