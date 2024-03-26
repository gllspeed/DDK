from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def alpha_beta_init_generate(mean, std):
    #np.random.seed(6)
    #data = np.random.normal(mean, std)
    data = 0
    while(data<=0):
        data = np.random.normal(mean, std)
    return torch.Tensor([data]).to(device)

def alpha_beta_add_noise(m, alpha=True, beta=True):
    if alpha:
        alpha_mean = float(m.alpha.data)
        alpha_std = 0.01873 * 5.1
        alpha_noise = alpha_beta_init_generate(alpha_mean, alpha_std)
        m.alpha.data = alpha_noise
    if beta:
        beta_mean = float(m.beta.data)
        beta_std = 0.05411 * 5.1
        beta_noise = alpha_beta_init_generate(beta_mean, beta_std)
        m.beta.data = beta_noise
###########映射方案一###############
def calcScaleZeroPoint(min_val, max_val, map_min, map_max):

    scale = float((max_val - min_val) / (map_max - map_min))

    zero_point = map_max - max_val / scale

    if zero_point < map_min:
        zero_point = map_min
    elif zero_point > map_max:
        zero_point = map_max

    zero_point = int(zero_point)

    return scale, zero_point


def mapping(x, scale, zero_point, map_min, map_max):

    q_x = zero_point + x * scale
    q_x.clamp_(map_min, map_max)

    return q_x

def de_mapping(q_x, scale, zero_point):
    return (q_x - zero_point)/scale

def weight_map(weight, noise_mean, noise_std, map_min, map_max):
    m, n = weight.shape
    ##########weight map G################
    weight_abs_max = torch.abs(weight).max()
    zero_point = int((map_max + map_min) / 2)
    scale = (map_max - zero_point) / weight_abs_max

    weight_q = mapping(weight, scale, zero_point, map_min, map_max)
    weight_q_flatten = weight_q.flatten()

    noise = np.random.normal(noise_mean, noise_std, weight_q_flatten.shape[0])
    weight_q_flatten_noise = weight_q_flatten + torch.Tensor(noise).to(device)
    weight_q_flatten_noise.clamp_(map_min, map_max)

    weight_noise = de_mapping(weight_q_flatten_noise, scale, zero_point)
    return torch.reshape(weight_noise, (m,n))

def w_add_noise(module, noise_mean, noise_std, map_min, map_max, clip_min=None, clip_max=None):
    w = module.weight.data
    #w.clamp_(clip_min, clip_max)
    w_noise = weight_map(w, noise_mean, noise_std, map_min, map_max)
    module.weight.data = w_noise