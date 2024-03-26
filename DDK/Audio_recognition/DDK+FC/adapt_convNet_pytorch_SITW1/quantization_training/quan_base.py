import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum

class Qmodes(Enum):
    #########layer_wise逐层量化， kernel_wise逐通道量化
    layer_wise=1
    kernel_wise=2

def get_default_kwargs_q(kwargs_q, layer_type):
    default = {'nbits':4}
    if isinstance(layer_type, Conv2dQ):
        default.update({'mode': Qmodes.layer_wise})
    elif isinstance(layer_type, LinearQ):
        pass
    elif isinstance(layer_type, ActQ):
        default.update({'signed':'Auto'})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q

class Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, **kwargs_q):
        super(Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=0, dilation=1, groups=1, bias=False)

        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = self.kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.q_mode = self.kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:
            self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits']=nbits

    def extra_repr(self):
        s_prefix = super(Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, **kwargs_q):
        super(LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class MaxpoolingQ(nn.MaxPool2d):
    def __init__(self, kernel_size=3, stride=1, padding=0, **kwargs_q):
        super(MaxpoolingQ, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(MaxpoolingQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        # self.signed = kwargs_q['signed']
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)