#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################
import logging
import os
import click

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

'''
@cli.command('test-model', short_help='Test a Keras model.')
@click.option('--working_dir', required=True, type=Ct.input_dir())
@click.option('--checkpoint_file', required=True, type=Ct.input_file())
def test_model(working_dir, checkpoint_file=None):
    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-softmax/ResCNN_checkpoint_102.h5
    # f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043

    # export CUDA_VISIBLE_DEVICES=0; python cli.py test-model
    # --working_dir /home/philippe/ds-test/triplet-training/
    # --checkpoint_file ../ds-test/checkpoints-triplets/ResCNN_checkpoint_175.h5
    # f-measure = 0.849, true positive rate = 0.798, accurechoacy = 0.997, equal error rate = 0.025

    test(working_dir, checkpoint_file)
'''
    #test(working_dir, checkpoint_file)
    #sys.exit(0)
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
    data_root = '/home/gaolili/deepLearning_project/action_classification_3Dconv/data'
    kx_train = np.load(os.path.join(data_root, 'train_npy', 'input_16_112_112_crop.npy'))
    kx_train = np.transpose(kx_train, (0, 4, 1, 2, 3))

    ky_train = np.load(os.path.join(data_root, 'train_npy', 'label_16_112_112_crop.npy'))

    kx_test = np.load(os.path.join(data_root, 'test_npy', 'input_16_112_112_crop.npy'))
    kx_test = np.transpose(kx_test, (0, 4, 1, 2, 3))

    ky_test = np.load(os.path.join(data_root, 'test_npy', 'label_16_112_112_crop.npy'))

    ###########dataset跟HF_V3的一样##############

    '''
    kx_train = np.load(os.path.join(data_root, 'train_npy', 'input_160_4.npy'))
    kx_train = np.reshape(kx_train, (kx_train.shape[0], kx_train.shape[1], 80, 60))
    kx_train = kx_train[:,np.newaxis, :,:,:]

    ky_train = np.load(os.path.join(data_root, 'train_npy', 'label_160_4.npy'))

    kx_test = np.load(os.path.join(data_root, 'test_npy', 'input_160_4.npy'))
    kx_test = np.reshape(kx_test, (kx_test.shape[0], kx_test.shape[1], 80, 60))
    kx_test = kx_test[:, np.newaxis, :, :, :]

    ky_test = np.load(os.path.join(data_root, 'test_npy', 'label_160_4.npy'))
    '''
    ###########随机打乱############################
    ###########dataset跟HF_V2的一样##############
    '''
    data_root = '/home/gaolili/deepLearning_project/action_classification_adapt_HF_V2/data'
    kx = np.load(os.path.join(data_root, 'input_160_4.npy'))
    ky = np.load(os.path.join(data_root, 'label_160_4.npy'))
    num_data = kx.shape[0]
    random.seed(100)
    random_index = [i for i in range(num_data)]
    random.shuffle(random_index)
    kx_random = kx[random_index, :, :]
    ky_random = ky[random_index, :]

    kx_train = kx_random[:int(num_data * 0.8), :, :]
    kx_train = np.reshape(kx_train, (kx_train.shape[0], kx_train.shape[1], 80, 60))
    kx_train = kx_train[:, np.newaxis, :, :, :]
    ky_train = ky_random[:int(num_data * 0.8), :]

    kx_test = kx_random[int(num_data * 0.8):, :, :]
    kx_test = np.reshape(kx_test, (kx_test.shape[0], kx_test.shape[1], 80, 60))
    kx_test = kx_test[:, np.newaxis, :, :, :]
    ky_test = ky_random[int(num_data * 0.8):, :]
    '''
    start_training(kx_train, ky_train, kx_test, ky_test, random_seed, trial)





if __name__ == '__main__':
    cli()


