
###########浮点转定点转整型的量化方案，scale正好是x的二次幂
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quan_base import Conv2dQ, Qmodes, LinearQ, ActQ


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def power_2(x):
     y = x.sign() *2 ** (x.abs().log2().round())
     y_grad = x
     return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w)
        if 'model_float' in kwargs:
            self.weight.data = torch.tensor(kwargs['model_float'])

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            ##########初始化scale##############################
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.

        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        #w_q = power_2((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def quantize_inference(self, x, qi_alpha=None, qo_alpha = None):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        alpha = 2**(self.alpha.abs().log2().round())
        #alpha = self.alpha
        #w_q = power_2((self.weight / alpha).clamp(Qn, Qp))
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp))
        x = F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if qi_alpha and qo_alpha:
            M = qi_alpha * alpha / qo_alpha
            x = M * x
            x.round_()
            x.clamp_(Qn, Qp).round_()
        return x

class LinearLSQ(LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w)
        if 'model_float' in kwargs:
            self.weight.data = torch.tensor(kwargs['model_float'])

    def forward(self, x, qi_alpha=None):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        bias_q = round_pass((self.bias / alpha / qi_alpha).clamp(Qn, Qp)) * alpha * qi_alpha
        #w_q = power_2((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        #return F.linear(x, w_q, self.bias)
        return F.linear(x, w_q, bias_q)

    def quantize_inference(self, x, qi_alpha = None, qo_alpha = None):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        #alpha = 2 ** (self.alpha.abs().log2().round())
        alpha = self.alpha
        #w_q = power_2((self.weight / alpha).clamp(Qn, Qp))
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp))
        if self.bias is not None and qi_alpha:
            bias_q = round_pass((self.bias / alpha / qi_alpha).clamp(Qn, Qp))
        else:
            bias_q = self.bias
        #x = F.linear(x, w_q, self.bias)
        x = F.linear(x, w_q, bias_q)
        if qi_alpha and qo_alpha:
            M = qi_alpha * alpha / qo_alpha
            x = M * x
            x.round_()
            x.clamp_(Qn, Qp).round_()
        return x


class ActLSQ(ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(ActLSQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return xnid
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        #x = power_2((x / alpha).clamp(Qn, Qp)) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return F.relu(x)

    def quantize_inference(self, x, qi_alpha=None, qo_alpha=None):

        return F.relu(x)

class SigmoidLSQ(ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(SigmoidLSQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return xnid
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        #x = power_2((x / alpha).clamp(Qn, Qp)) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return F.sigmoid(x)

    def quantize_inference(self, x, qi_alpha=None, qo_alpha=None):
        #alpha = 2 ** (self.alpha.abs().log2().round())
        alpha = self.alpha
        return F.sigmoid(x * alpha)


class SigmoidLSQ_quantize(ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(SigmoidLSQ_quantize, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return xnid
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        #######x clip到[-8,8]
        x.clamp(-8,8)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        #x = power_2((x / alpha).clamp(Qn, Qp)) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x

    def quantize_inference(self, x, qi_alpha=None, qo_alpha=None):
        #alpha = 2 ** (self.alpha.abs().log2().round())
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        alpha = self.alpha
        x = x * alpha
        x.clamp(-8, 8)
        x = round_pass((x / alpha).clamp(Qn, Qp))
        return x


class InputLSQ(ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(InputLSQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return xnid
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        #x = power_2((x / alpha).clamp(Qn, Qp)) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x

    def quantize_inference(self, x, qi_alpha=None, qo_alpha=None):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        # if x.min() < -1e-5:
        #    Qn = -2 ** (self.nbits - 1)
        #    Qp = 2 ** (self.nbits - 1) - 1
        # else:
        #    Qn = 0
        #    Qp = 2 ** self.nbits - 1

        #alpha = 2 ** (self.alpha.abs().log2().round())
        alpha = self.alpha
        #x = power_2((x / alpha).clamp(Qn, Qp))
        x = round_pass((x / alpha).clamp(Qn, Qp))
        x_float = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        return x, x_float