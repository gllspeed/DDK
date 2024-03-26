import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import torchvision
import matplotlib.pyplot as plt
from model import Net
import numpy as np
import sklearn.metrics as sm

def test(model, device, test_loader, num_data, num_class):
    test_num = int(num_data * 0.15)
    model.eval()
    test_loss = 0
    correct = 0

    targets = []
    preds = []
    class_total = [0] * num_class
    true_class = [0] * num_class

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        #print("test_data", data)
        #print("test_label", target)
        output,target = model(data, target)
        #print("test_label", target)
        #output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()


    print('\nTest set: Accuracy: {:.0f}%\n'.format(
        100. * correct / test_num
    ))
    return accuracy

if __name__ == "__main__":
    input_data = 'DDK'#########'DDK
    if input_data=='DDK':
        batch_size = 1000#372
        test_batch_size = 1000#372
        num_data = 1000
        input_mat = 'speaker_no_fft_norm.mat'
        label_mat = 'targets_no_fft_norm.mat'
    elif input_data=='fft_DDK':
        batch_size = 3120  # 372
        test_batch_size = 3120  # 372
        num_data = 3120
        input_mat = 'speaker_fft14_norm.mat'
        label_mat = 'targets_fft14_norm.mat'

    num_class = 10
    epochs = 1000
    lr = 0.01
    momentum = 0.5
    save_model = False
    using_bn = False


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###########################训练audio##################
    acc_best = 0
    #for seed in range(1):
    #seed = 10#######DDK处理当作数据预处理，效果最好
    seed = 100
    print("i", seed)

    data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file'
    #data_path = 'D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\DDK'
    data = dataset.load_voice_print(data_path, input_mat, label_mat, training='is_training',sed=seed)
    Loader = dataset.generate_dataLoader(data, batch_size=batch_size)

    model = Net(num_class=num_class, num_data=num_data).to(device)
    if input_data == 'DDK':
        model.load_state_dict(torch.load(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_speaker/ckpt/code_optim/adapt_speaker_DDK_0.9733333333333334.pt'))

        for name, parameter in model.named_parameters():
            print(name)
            print(parameter)

    elif input_data == 'fft_DDK':
        model.load_state_dict(torch.load(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/fft_14_adapt_convNet_pytorch_speaker/ckpt/nfft14/adapt_speaker_fft14_0.967948717948718.pt'))
    accuracy = test(model, device, Loader, num_data, num_class)



