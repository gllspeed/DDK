from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class Mnist_Dataset(Dataset):
    def __init__(self, kx, ky, transform=None):
        self.kx = kx
        self.ky = ky
        self.transform = transform


    def __getitem__(self, index):

        data = self.kx[index,:]
        label = int(self.ky[0][index])
        if self.transform is not None:
            data = self.transform(data)
        else:
            data = np.array(data, dtype='float32')
            data = torch.tensor(data)
        return data, label

    def __len__(self):
        return len(self.kx)