from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class AudioDataset(Dataset):
    def __init__(self, kx, ky, transform=None):
        self.kx = kx
        self.ky = ky
        self.transform = transform


    def __getitem__(self, index):
    ###########audio_data shape:[257,160，1]###########前期做HF
        audio = self.kx[index,:]
        label = int(self.ky[index][0])
        if self.transform is not None:
            audio = self.transform(audio)
        else:
            audio = np.array(audio, dtype='float32')
            audio = torch.tensor(audio)
        return audio, label

    def __len__(self):
        return len(self.kx)