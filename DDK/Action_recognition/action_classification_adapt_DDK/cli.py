#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################
import logging
import os
import click
import torch
#from test import test
from train import start_training
from utils import ClickType as Ct, ensures_dir
from utils import init_pandas
import numpy as np
import sys
import random
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

    ############计算train, test数据集###########################
    is_DDK = True
    data_root = '/home/gaolili/deepLearning_project/action_classification_adapt_DDK_V4_new/data'
    kx_train = np.load(os.path.join(data_root, 'train_npy', 'input_16_4_shuffle.npy'))
    ky_train = np.load(os.path.join(data_root, 'train_npy', 'label_16_4_shuffle.npy'))
    kx_test = np.load(os.path.join(data_root, 'test_npy', 'input_16_4_shuffle.npy'))
    ky_test = np.load(os.path.join(data_root, 'test_npy', 'label_16_4_shuffle.npy'))

    ##############打乱训练样本#######################
    random.seed(100)
    random_index = [i for i in range(kx_train.shape[0])]
    random.shuffle(random_index)
    kx_train = kx_train[random_index, :]
    ky_train = ky_train[random_index]
    #torch.manual_seed(trial)
    start_training(kx_train, ky_train, kx_test, ky_test, random_seed, trial, is_DDK)


if __name__ == '__main__':
    cli()


