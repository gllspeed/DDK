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
    def __init__(self, dataloader_path, input_mat,num_class, num_every_class, training,  sed):

            data = scio.loadmat(os.path.join(dataloader_path, input_mat))

            inputs = data['whaleInputNorm']
            labels = data['whaleTargetNorm']

            input_array = np.transpose(inputs)
            label_array = np.transpose(labels)
            label_list = [i.index(1) for i in label_array.tolist()]
            label = np.array(label_list)
            
            random.seed(sed)
            random_index = [i for i in range(input_array.shape[0])]
            random.shuffle(random_index)
            input_array = input_array[random_index, :]
            label = label[random_index]
            
            input_train = input_array[0:int(input_array.shape[0]*0.8),:]
            label_train = label[0:int(input_array.shape[0]*0.8)]
            
            input_test = input_array[int(input_array.shape[0]*0.8):,:]
            label_test = label[int(input_array.shape[0]*0.8):]
            

            if training == 'is_training':
                self.inputs = torch.tensor(input_train)
                self.labels = torch.tensor(label_train)
            else:
                self.inputs = torch.tensor(input_test)
                self.labels = torch.tensor(label_test)

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
def matrix_norm(input):
    input_norm = (input - input.min()) / (input.max() - input.min())
    return input_norm
