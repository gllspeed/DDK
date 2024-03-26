#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################
import logging
import os
import click
import random
#from test import test
from train import start_training
from utils import ClickType as Ct, ensures_dir, read_mat
from utils import init_pandas
import numpy as np
import sys
import torch
logger = logging.getLogger(__name__)
VERSION = '3.0a'


@click.group()
def cli():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    init_pandas()

@cli.command('version', short_help='Prints the version.')
def version():
    print(f'Version is {VERSION}.')


@cli.command('train-model', short_help='Train a Keras model.')
@click.option('--random_seed', default=1, show_default=True, type=int)
@click.option('--trial', default=1, show_default=True, type=int)
def train_model(random_seed, trial):
    # PRE TRAINING

    # commit a5030dd7a1b53cd11d5ab7832fa2d43f2093a464
    # Merge: a11d13e b30e64e
    # Author: Philippe Remy <premy.enseirb@gmail.com>
    # Date:   Fri Apr 10 10:37:59 2020 +0900
    # LibriSpeech train-clean-data360 (600, 100). 0.985 on test set (enough for pre-training).

    # TRIPLET TRAINING
    # [...]
    # Epoch 175/1000
    # 2000/2000 [==============================] - 919s 459ms/step - loss: 0.0077 - val_loss: 0.0058
    # Epoch 176/1000
    # 2000/2000 [==============================] - 917s 458ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 177/1000
    # 2000/2000 [==============================] - 927s 464ms/step - loss: 0.0075 - val_loss: 0.0059
    # Epoch 178/1000
    # 2000/2000 [==============================] - 948s 474ms/step - loss: 0.0073 - val_loss: 0.0058
    print("random_seed", random_seed)
    ############计算train, test数据集###########################
    is_DDK = True
    mat_path = '/home/gaolili/data/ModelNet10/ModelNet10_voxelized_mat'
    label_map = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7,
                 'table': 8, 'toilet': 9}
    kx_train, ky_train, kx_test, ky_test = read_mat(mat_path, label_map)
    ##############打乱训练样本#######################

    random.seed(100)
    random_index = [i for i in range(kx_train.shape[0])]
    random.shuffle(random_index)
    kx_train = kx_train[random_index, :]
    ky_train = ky_train[random_index]

    ##############alpha，beta指定值初始化###############################
    '''
    alpha_init_path = '/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK/checkpoints/DDK/loss_accuracy/7/499_alpha.txt'
    beta_init_path = '/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK/checkpoints/DDK/loss_accuracy/7/499_beta.txt'
    alpha_init = open(alpha_init_path, "r").readlines()[0].split(' ')
    beta_init = open(beta_init_path, "r").readlines()[0].split(' ')
    alpha_init = torch.tensor([float(alpha) for alpha in alpha_init])
    beta_init = torch.tensor([float(beta) for beta in beta_init])
    '''
    #torch.manual_seed(trial)
    start_training(kx_train, ky_train, kx_test, ky_test, random_seed, trial, is_DDK)


if __name__ == '__main__':
    cli()


