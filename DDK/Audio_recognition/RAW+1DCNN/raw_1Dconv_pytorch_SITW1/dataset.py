import os
import json
import pickle
import scipy.io as scio

import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from PIL import ImageEnhance
import random
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#############add voice####################
class load_voice_print:
    def __init__(self, dataloader_path, training, sed):
            #if dataloader_path.endswith('mat'):

            #data = scio.loadmat(dataloader_path)
            #inputs = data['whaleInputSpectro']
            #labels = data['whaleTargetSpectro']
            data = scio.loadmat(os.path.join(dataloader_path, 'speaker_Raw_1Dconv.mat'))
            target = scio.loadmat(os.path.join(dataloader_path, 'targets_Raw_1Dconv.mat'))
            inputs = data['samples']
            labels = target['targets']
            #######归一化到-1~1##############
            #####归一化公式y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin#######
            #inputs = 2 * inputs -1
            input_list = np.transpose(inputs).tolist()
            label_array = np.transpose(labels)
            label_list = [i.index(1) for i in label_array.tolist()]

            random.seed(sed)
            random_index = [i for i in range(len(input_list))]
            random.shuffle(random_index)
            if training == 'is_training':
                random_index_train = random_index[:int(len(input_list)*0.8)]
                input_train = torch.tensor([input_list[index] for index in random_index_train])
                label_train = torch.tensor([label_list[index] for index in random_index_train])
                self.inputs = input_train
                self.labels = label_train
            if training == 'is_val':
                random_index_val = random_index[int(len(input_list)*0.7):int(len(input_list)*0.85)]
                input_val = torch.tensor([input_list[index] for index in random_index_val])
                label_val = torch.tensor([label_list[index] for index in random_index_val])
                self.inputs = input_val
                self.labels = label_val
            if training == 'is_test':
                random_index_test = random_index[int(len(input_list)*0.8):]
                input_test = torch.tensor([input_list[index] for index in random_index_test])
                label_test = torch.tensor([label_list[index] for index in random_index_test])
                self.inputs = input_test
                self.labels = label_test

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return input, label

    def __len__(self):
        return len(self.inputs)

def generate_dataLoader(data, batch_size = 32):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return data_loader

########################################
def load_data(dataloader_path):
    print('加载dataloader.pth...')
    with open(dataloader_path, 'rb') as f:
        train_dataloader = pickle.load(f)
        val_dataloader = pickle.load(f)
    print('加载完成...')
    return train_dataloader, val_dataloader


def save_dataloader(train_dataloader, val_dataloader, dataloader_path):
    print('创建dataloader.pth...')
    with open(dataloader_path, 'wb') as f:
        pickle.dump(train_dataloader, f)
        pickle.dump(val_dataloader, f)
    print('保存完成...')


