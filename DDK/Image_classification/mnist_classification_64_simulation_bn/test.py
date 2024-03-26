import logging
import sys
import numpy as np
from tqdm import tqdm

from model import *
from torchvision import datasets, transforms
import os
import torch
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

def test_add_Noise(model, device, test_loader, num_class, noise_mean = None, noise_std = None, map_min = None, map_max = None):
    model.eval()
    test_loss = 0
    correct = 0
    w_add_noise(model.fc1, noise_mean, noise_std, map_min, map_max)
    w_add_noise(model.fc, noise_mean, noise_std, map_min, map_max)
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

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/home/gaolili/deepLearning_project/data/mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
                #transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
        )


    model = Net().to(device)
    #############每组实验，最好的三个模型的融合##########################
    accuracy_all = []

    for trial in range(10):##########表示五组实验
        ###############加噪声训练的模型权重#############

        pretrained_state = torch.load(
            '/home/gaolili/deepLearning_project/mnist_classification_64_simulation_bn/ckpt_DDK_fc_bn_relu/mnist_14_0.973.pt')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        model.load_state_dict(model_dict)

        alpha_beta_add_noise(model.DDK)
        #accuracy = test(model, device, test_loader, num_class)

        accuracy = test_add_Noise(model, device, test_loader, num_class, noise_mean=0.20329,
                                         noise_std=1.14726,
                                         map_min=40,
                                         map_max=300)

        accuracy_all.append(accuracy[-1])


    accuracy_best_max = max(accuracy_all)
    accuracy_best_mean = np.mean(accuracy_all)
    accuracy_best_std = np.std(accuracy_all)
    print("accuracy_all",accuracy_all)
    print("accuracy_best_max", accuracy_best_max)
    print("accuracy_best_mean", accuracy_best_mean)
    print("accuracy_best_std", accuracy_best_std)
