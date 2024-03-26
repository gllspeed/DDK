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
from constants import TRAIN_TEST_RATIO
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

def read_mat(mat_path, label_map):
    kx = []
    ky = []
    kx_train = []
    ky_train = []
    kx_test = []
    ky_test = []
    examples_every_class = 100
    for root, dirs, _ in os.walk(mat_path):
        for dir in dirs:
            #print(dir)
            if dir in label_map:
                label = label_map[dir]
                kx_temp = []
                ky_temp = []
                for train_or_test in ['train']:
                    object_path = os.path.join(root, dir, train_or_test)
                    for object_name in os.listdir(object_path):
                        if 'night_stand' in object_name:
                            #print(object_name)
                            temp = int(object_name.split('_')[2])
                        else:
                            temp = int(object_name.split('_')[1])
                        if temp <= examples_every_class:
                            object_data = scio.loadmat(os.path.join(object_path, object_name))['instance']###########(Y,X,Z)

                            kx_temp.append(object_data.tolist())

                kx_temp = np.array(kx_temp)
                kx_temp_train = kx_temp[:int(kx_temp.shape[0] * 0.8), :, :, :].tolist()
                kx_temp_test = kx_temp[int(kx_temp.shape[0] * 0.8):, :, :, :].tolist()
                kx_train.extend(kx_temp_train)
                ky_train.extend([label] * len(kx_temp_train))
                kx_test.extend(kx_temp_test)
                ky_test.extend([label] * len(kx_temp_test))

    kx_train = np.array(kx_train)
    ky_train = np.array(ky_train)
    kx_test = np.array(kx_test)
    ky_test = np.array(ky_test)

    kx_train = kx_train[:,np.newaxis, :, :, :]
    ky_train = ky_train[:, np.newaxis]
    kx_test = kx_test[:, np.newaxis, :, :, :]
    ky_test = ky_test[:, np.newaxis]

    random.seed(100)
    random_index = [i for i in range(kx_train.shape[0])]
    random.shuffle(random_index)
    kx_train = kx_train[random_index, :, :, :, :]
    ky_train = ky_train[random_index, :]

    return kx_train, ky_train, kx_test, ky_test


'''
###########生成数据集#########################
def read_mat(mat_path, label_map):
    kx = []
    ky = []
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
                            print(object_name)
                            temp = int(object_name.split('_')[2])
                        else:
                            temp = int(object_name.split('_')[1])
                        if temp <= examples_every_class:
                            object_data = scio.loadmat(os.path.join(object_path, object_name))['instance']###########(Y,X,Z)

                            kx.append(object_data.tolist())
                            ky.append([label])

    kx = np.array(kx)
    ky = np.array(ky)

    random.seed(100)
    random_index = [i for i in range(kx.shape[0])]
    random.shuffle(random_index)
    kx = kx[random_index, :, :, :]
    kx = kx[:,np.newaxis, :, :, :]
    ky = ky[random_index,:]
    kx_train = kx[:int(kx.shape[0] * 0.8), :, :, :, :]
    ky_train = ky[:int(kx.shape[0] * 0.8), :]

    kx_test = kx[int(kx.shape[0] * 0.8):, :, :, :, :]
    ky_test = ky[int(ky.shape[0] * 0.8):, :]

    return kx_train, ky_train, kx_test, ky_test
'''
if __name__ =='__main__':
    #######################直接生成kx, ky太慢，打算先获取RGB图片，目前还没调试################################
    mat_path = '/home/gaolili/data/ModelNet10/ModelNet10_voxelized_mat'

    label_map = {'bathtub':0, 'bed':1, 'chair':2, 'desk':3, 'dresser':4, 'monitor':5, 'night_stand':6, 'sofa':7,
                 'table':8, 'toilet':9}
    read_mat(mat_path, label_map)