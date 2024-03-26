import logging
import os
import random
import shutil
from glob import glob
import cv2
import click
import dill
import numpy as np
import pandas as pd
import scipy.io as scio
from natsort import natsorted
from collections import defaultdict
from constants import num_frames
import torch
from torch.optim.lr_scheduler import _LRScheduler
logger = logging.getLogger(__name__)


def find_files(directory, ext='wav'):
    return sorted(glob(directory + f'/**/*.{ext}', recursive=True))


def init_pandas():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def create_new_empty_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def ensure_dir_for_filename(filename: str):
    ensures_dir(os.path.dirname(filename))


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


class ClickType:

    @staticmethod
    def input_file(writable=False):
        return click.Path(exists=True, file_okay=True, dir_okay=False,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def input_dir(writable=False):
        return click.Path(exists=True, file_okay=False, dir_okay=True,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def output_file():
        return click.Path(exists=False, file_okay=True, dir_okay=False,
                          writable=True, readable=True, resolve_path=True)

    @staticmethod
    def output_dir():
        return click.Path(exists=False, file_okay=False, dir_okay=True,
                          writable=True, readable=True, resolve_path=True)


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned


def load_best_checkpoint(checkpoint_dir):
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    if len(checkpoints) != 0:
        return checkpoints[-1]
    return None


def delete_older_checkpoints(checkpoint_dir, max_to_keep=5):
    assert max_to_keep > 0
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    checkpoints_to_keep = checkpoints[-max_to_keep:]
    for checkpoint in checkpoints:
        if checkpoint not in checkpoints_to_keep:
            os.remove(checkpoint)


def enable_deterministic():
    print('Deterministic mode enabled.')
    np.random.seed(123)
    random.seed(123)


def load_pickle(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading PKL file: {file}.')
    with open(file, 'rb') as r:
        return dill.load(r)


def load_npy(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading NPY file: {file}.')
    return np.load(file)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

###########按每个3D图片的特征维度做DDK操作########################
'''
def DDK(input, a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.transpose(torch.flatten(input, start_dim=1), 0,1)
    n, m = input.shape
    num_data = int(m / num_frames)
    input = torch.reshape(input, (n, num_data, num_frames))

    weight_matrix = torch.ones(n, num_data, num_frames).to(device)
    for k in range(1, num_frames):
        temp = 1 / weight_matrix[:,:, k - 1]
        weight_matrix[:,:, k] = weight_matrix[:,:, k - 1] + a[k] * temp - b[k] * input[:,:, k - 1]
    weight_matrix = torch.reshape(weight_matrix, (n, m))
    weight_matrix = torch.transpose(weight_matrix, 0, 1)
    return weight_matrix
'''


def DDK(input, a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l, n, m = input.shape
    # input = torch.transpose(input, 2, 1)

    weight_matrix = torch.ones(l, n, m).to(device)
    for k in range(1, m):
        temp = 1 / weight_matrix[:, :, k - 1]
        weight_matrix[:, :, k] = weight_matrix[:, :, k - 1] + a[k] * temp - b[k] * input[:, :, k - 1]
    weight_matrix = torch.flatten(weight_matrix, start_dim=1)
    return weight_matrix

'''
##########按所有样本做DDK##################
def DDK(input, a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.transpose(input, 0, 1).to(device)
    m, n, = input.shape
    weight_matrix = torch.ones(m, n).to(device)
    for k in range(1, n):

        temp = 1 / weight_matrix[:,k-1]
        weight_matrix[:, k] = weight_matrix[:, k - 1] + a *temp - b * input[:, k-1]

    weight_matrix = torch.transpose(weight_matrix, 0, 1)

    return weight_matrix
'''


###########生成数据集#########################
def read_mat(mat_path, label_map):
    kx_train = []
    ky_train = []
    kx_test = []
    ky_test = []
    is_read = 0
    id = 0
    examples_every_class = 100
    for root, dirs, _ in os.walk(mat_path):
        for dir in dirs:
            #print(dir)
            if dir in label_map:
                label = label_map[dir]
                kx_temp = []
                for train_or_test in ['train']:
                    object_path = os.path.join(root, dir, train_or_test)
                    for object_name in os.listdir(object_path):
                        if 'night_stand' in object_name:
                            temp = int(object_name.split('_')[2])
                        else:
                            temp = int(object_name.split('_')[1])
                        if temp <= examples_every_class:
                            #object_data = scio.loadmat(os.path.join(object_path, object_name))['instance'].transpose(1,0,2)###########(Y,X,Z)
                            object_data = scio.loadmat(os.path.join(object_path, object_name))['instance']
                            #object_data = np.reshape(object_data, (32, -1))
                           
                            kx_temp.extend(object_data.tolist())
                kx_temp = np.array(kx_temp)

                kx_temp_train = kx_temp[:int(kx_temp.shape[0] * 0.8), :].tolist()
                kx_temp_test = kx_temp[int(kx_temp.shape[0] * 0.8):, :].tolist()
                kx_train.extend(kx_temp_train)
                ky_train.extend([label] * len(kx_temp_train))
                kx_test.extend(kx_temp_test)
                ky_test.extend([label] * len(kx_temp_test))

    kx_train = np.array(kx_train)
    ky_train = np.array(ky_train)
    kx_test = np.array(kx_test)
    ky_test = np.array(ky_test)

    ky_train = ky_train[:, np.newaxis]
    ky_test = ky_test[:, np.newaxis]
    return kx_train, ky_train, kx_test, ky_test


'''
###########生成数据集#########################
def read_mat(mat_path, label_map):
    kx = []
    ky = []
    kx_train = []
    ky_train = []
    kx_test = []
    ky_test = []
    is_read = 0
    id = 0
    examples_every_class = 100
    for root, dirs, _ in os.walk(mat_path):
        for dir in dirs:
            #print(dir)
            if dir in label_map:
                label = label_map[dir]

                for train_or_test in ['train']:
                    object_path = os.path.join(root, dir, train_or_test)
                    for object_name in os.listdir(object_path):
                        if 'night_stand' in object_name:
                            temp = int(object_name.split('_')[2])
                        else:
                            temp = int(object_name.split('_')[1])
                        if temp <= examples_every_class:
                            #object_data = scio.loadmat(os.path.join(object_path, object_name))['instance'].transpose(1,0,2)###########(Y,X,Z)
                            object_data = scio.loadmat(os.path.join(object_path, object_name))['instance']
                            #object_data = np.reshape(object_data, (32, -1))

                            kx.append(object_data.tolist())
                            ky.append([label]*32)
    kx = np.array(kx)
    ky = np.array(ky)

    random.seed(100)
    random_index = [i for i in range(kx.shape[0])]
    random.shuffle(random_index)
    kx = kx[random_index, :, :, :]
    ky = ky[random_index, :]
    kx_train = kx[:int(kx.shape[0] * 0.8), :, :, :]
    ky_train = ky[:int(kx.shape[0] * 0.8), :]

    kx_test = kx[int(kx.shape[0] * 0.8):, :, :, :]
    ky_test = ky[int(ky.shape[0] * 0.8):, :]


    kx_train = np.array(np.reshape(kx_train, (-1, 32, 32)))
    ky_train = np.array(np.reshape(ky_train, (-1, 1)))
    kx_test = np.array(np.reshape(kx_test, (-1, 32, 32)))
    ky_test = np.array(np.reshape(ky_test, (-1, 1)))

    return kx_train, ky_train, kx_test, ky_test

'''
#####################噪声添加###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def alpha_beta_init_generate(mean, std):
    #np.random.seed(6)
    #data = np.random.normal(mean, std)
    data = 0
    while(data<=0):
        data = np.random.normal(mean, std)
    return data

def alpha_beta_add_noise(m, alpha=True, beta=True):
    if alpha:

        alpha_mean = m.alpha.data.cpu().numpy()
        alpha_noise_all = torch.empty(alpha_mean.shape[0]).to(device)
        alpha_std = 0.01873 * 5.1
        for i in range(alpha_mean.shape[0]):
            alpha_noise = alpha_beta_init_generate(alpha_mean[i], alpha_std)
            alpha_noise_all[i] = alpha_noise
        m.alpha.data = alpha_noise_all


    if beta:
        beta_mean = m.beta.data.cpu().numpy()
        beta_noise_all = torch.empty(beta_mean.shape[0]).to(device)
        beta_std = 0.05411 * 5.1
        for i in range(beta_mean.shape[0]):
            beta_noise = alpha_beta_init_generate(beta_mean[i], beta_std)
            beta_noise_all[i] = beta_noise
        m.beta.data = beta_noise_all
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

def w_add_noise(module, noise_mean, noise_std, map_min, map_max):
    w = module.weight.data
    #w.clamp_(-1, 1)
    w_noise = weight_map(w, noise_mean, noise_std, map_min, map_max)
    module.weight.data = w_noise

if __name__ =='__main__':
    #######################直接生成kx, ky太慢，打算先获取RGB图片，目前还没调试################################
    mat_path = '/home/gaolili/data/ModelNet10/ModelNet10_voxelized_mat'

    label_map = {'bathtub':0, 'bed':1, 'chair':2, 'desk':3, 'dresser':4, 'monitor':5, 'night_stand':6, 'sofa':7,
                 'table':8, 'toilet':9}
    read_mat(mat_path, label_map)