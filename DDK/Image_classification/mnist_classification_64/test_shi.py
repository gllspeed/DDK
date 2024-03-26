import logging
import sys
import numpy as np
from tqdm import tqdm

from model import *
from torchvision import datasets, transforms
import os
import torch
from dataset_shi import *
import scipy.io as scio
logger = logging.getLogger(__name__)


def test(model, device, test_loader, num_class):
    model.eval()
    test_loss = 0
    correct = 0

    targets = []
    preds = []
    class_total = [0] * num_class
    true_class = [0] * num_class
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    #############计算每一类的精度################
    for index in range(len(targets)):
        class_total[targets[index]] += 1
        if preds[index] == targets[index]:
            true_class[preds[index]] += 1

    accuracy = [true_class[index] / class_total[index] for index in range(num_class)]
    accuracy_avg = correct / len(test_loader.dataset)
    accuracy.append(accuracy_avg)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))
    return accuracy





if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_batch_size = 64

    num_class = 10

    is_DDK = True

    data = scio.loadmat('/home/gaolili/deepLearning_project/mnist_classification_32/mnist_data.mat')
    input = data['data']
    input = input[:,np.newaxis, :,:]
    label = data['label']
    test_data = Mnist_Dataset(input, label)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=1,
                                              pin_memory=True)
    model = Net().to(device)
    model.load_state_dict(torch.load(
        '/home/gaolili/deepLearning_project/mnist_classification_64/mnist_1_0.9649.pt'))
    accuracy = test(model, device, test_loader, num_class)

    #############每组实验，最好的三个模型的融合##########################
    accuracy_best = []
    accuracy_fusion_avg_all = []
    for trial in ['1', '2', '3', '4', '5']:##########表示五组实验
        accuracy_path = os.path.join('/home/gaolili/deepLearning_project/mnist_classification_64/result/DDK_0_1',trial)
        accuracy_epochs = open(os.path.join(accuracy_path, 'accuracy.txt'), 'r')
        accuracy_everyclass = []
        accuracy_avg = []
        accuracy_all = []
        accuracy_fusion = [0] * num_class
        for line in accuracy_epochs.readlines():
            data = line.split('\n')[0].split(' ')
            data = [float(i) for i in data]
            accuracy_avg.append(data[-1])
            accuracy_everyclass.append(data[0:-1])
        accuracy_avg_sort = sorted(accuracy_avg)
        accuracy_avg_max_3 = accuracy_avg_sort[-3:]
        accuracy_avg_max_3_index = [accuracy_avg.index(accuracy_temp) for accuracy_temp in accuracy_avg_max_3]
        for index in accuracy_avg_max_3_index:
            accuracy_all.append(accuracy_everyclass[index])
        for index in range(num_class):
            accuracy_fusion[index] = max(accuracy_all[0][index], accuracy_all[1][index], accuracy_all[2][index])
        accuracy_fusion_avg = np.mean(accuracy_fusion)
        accuracy_fusion_avg_all.append(accuracy_fusion_avg)







    accuracy_best_max = max(accuracy_fusion_avg_all)
    accuracy_best_mean = np.mean(accuracy_fusion_avg_all)
    accuracy_best_std = np.std(accuracy_fusion_avg_all)
    print("accuracy_best_max", accuracy_best_max)
    print("accuracy_best_mean", accuracy_best_mean)
    print("accuracy_best_std", accuracy_best_std)
