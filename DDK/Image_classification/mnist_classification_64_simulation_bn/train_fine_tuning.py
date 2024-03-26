from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import numpy as np
from utils import *
def train(model, device, train_loader, optimizer, epoch, noise_mean = None, noise_std = None, map_min = None, map_max = None):
    model.train()
    print("model.DDK.ALPHA", model.DDK.alpha)
    print("model.DDK.BETA", model.DDK.beta)

    ################################################
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx % 50 == 0:
        #    for m in model.modules():
        #        if isinstance(m, nn.Linear):
        #            w_add_noise(m, noise_mean, noise_std, map_min, map_max)
        ############一个epoch, 参数添加噪声####################
        #for m in model.modules():
        #    if isinstance(m, nn.Linear):
        #        w_add_noise(m, noise_mean, noise_std, map_min, map_max)
        #    if isinstance(m, DDK):
        #        alpha_beta_add_noise(m, alpha=False)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()
        #for name, parameters in model.named_parameters():
        #    if name == 'DDK.alpha':
        #        parameters.data.clamp_(0.001, 1)
        #    if name == 'DDK.beta':
        #        #parameters.data.clamp_(0.3, 2)
        #        parameters.data.clamp_(0.001, 1)
        if batch_idx % 50 == 0:

            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))



def test(model, device, test_loader, num_class, epoch, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300):
    model.eval()
    test_loss = 0
    correct = 0

    ############一个epoch, 参数添加噪声, alpha, beta写进去已固定，
    # 边训练，边测试的时候（添加噪声测试），在测试完进行下一次的训练时使用的添加了噪声的权重进行训练，相当于给训练集的第一个batch添加了噪声##
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w_add_noise(m, noise_mean, noise_std, map_min, map_max)

    ############################################

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
    for trial in range(51,52):
        batch_size = 64
        test_batch_size = 64
        seed = 1
        epochs = 50
        lr = 0.01
        momentum = 0.5
        save_model = True
        using_bn = True
        num_class = 10
        input_data_DDK = True

        torch.manual_seed(trial)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/home/gaolili/deepLearning_project/data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Normalize((0.1307,), (0.3081,))
                                #transforms.Normalize((0.5,), (0.5,))
                           ])),
            batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/home/gaolili/deepLearning_project/data/mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
                #transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
        )

        model = Net().to(device)

        model.load_state_dict(torch.load('/home/gaolili/deepLearning_project/mnist_classification_64_simulation/pretrained_model/mnist_1_0.9647.pt'))
        ############alpha, beta的值添加噪声###########
        for m in model.modules():
            if isinstance(m, DDK):
                alpha_beta_add_noise(m)
        ###########直接测试###########
        accuracy = test(model, device, test_loader, num_class, 1, noise_mean=0.20329, noise_std=1.14726, map_min=40,
                        map_max=300)
        print("pretrained_accuracy", accuracy[-1])
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


        #optimizer = optim.Adam(model.parameters(), lr=lr)

        test_accuracy = []
        acc_best = 0
        acc_best_epoch = 0
        alphas = []
        betas = []

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300)

            alphas.append(model.DDK.alpha.cpu().detach().numpy()[0])
            betas.append(model.DDK.beta.cpu().detach().numpy()[0])
            accuracy = test(model, device, test_loader, num_class, epoch, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300)
            test_accuracy.append(accuracy)
            if acc_best_epoch < accuracy[-1]:
                acc_best_epoch = accuracy[-1]
                if acc_best_epoch>=0.94:
                    if input_data_DDK:
                        torch.save(model.state_dict(),
                                   '/home/gaolili/deepLearning_project/mnist_classification_64_simulation/ckpt_DDK/mnist_' + str(trial)+'_'+ str(
                                       acc_best_epoch) + '.pt')
                    else:
                        torch.save(model.state_dict(),
                                   '/home/gaolili/deepLearning_project/mnist_classification_64_simulation/ckpt/mnist_' + str(trial)+'_'+ str(
                                       acc_best_epoch) + '.pt')

        if acc_best_epoch > acc_best:
            acc_best = acc_best_epoch



        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/mnist_classification_64_simulation/result/DDK' ,str(trial), 'accuracy.txt'), test_accuracy, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/mnist_classification_64_simulation/result/DDK',str(trial), 'alpha.txt'), alphas, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/mnist_classification_64_simulation/result/DDK',str(trial), 'beta.txt'), betas, fmt='%.6f')

        print("acc_best", acc_best)
