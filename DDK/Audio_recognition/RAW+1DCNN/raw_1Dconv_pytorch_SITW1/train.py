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
from thop import clever_format
from thop import profile

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    total_loss = 0
    count = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        count +=1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
    print('\nTrain set: Accuracy: {:.6f}%\n'.format(100. * correct / len(train_loader.dataset)))
    avg_loss = total_loss/count
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    preds = []
    class_total = [0] * 10
    true_class = [0] * 10
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
    accuracy = [true_class[index] / class_total[index] for index in range(4)]
    accuracy_avg = correct / len(test_loader.dataset)
    accuracy.append(accuracy_avg)

    precision = sm.precision_score(targets, preds, average=None).tolist()
    precision_avg = np.mean(precision)
    precision.append(precision_avg)

    recall = sm.recall_score(targets, preds, average=None).tolist()
    recall_avg = np.mean(recall)
    recall.append(recall_avg)

    F1_score = sm.f1_score(targets, preds, average=None).tolist()
    F1_score_avg = np.mean(F1_score)
    F1_score.append(F1_score_avg)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))
    return test_loss, accuracy, precision, recall, F1_score

def cal_flops(model):
    input = torch.randn(1, 1600).to(device)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    a = 0


if __name__ == "__main__":
    for trial in range(1,6):
        batch_size = 64
        test_batch_size = 64

        epochs = 100
        lr = 0.01
        momentum = 0.5
        save_model = False
        using_bn = False


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ###########################训练audio##################
        acc_best = 0
        #for seed in range(1):
        seed=10
        print("i", seed)
        #data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/WhaleSong_fft.mat'
        #data_path = 'D:\\Voiceprint_20201027\\WhaleSong_fft.mat'
        data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file'
        #data_path = 'D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\fft'
        train_data = dataset.load_voice_print(data_path, training='is_training', sed=seed)
        val_data = dataset.load_voice_print(data_path, training='is_val', sed=seed)
        test_data = dataset.load_voice_print(data_path, training='is_test', sed=seed)
        train_Loader = dataset.generate_dataLoader(train_data, batch_size=batch_size)
        val_Loader = dataset.generate_dataLoader(val_data, batch_size=batch_size)
        test_Loader = dataset.generate_dataLoader(test_data, batch_size=batch_size)

        model = Net().to(device)
        cal_flops(model)
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        losses = []
        accs = []
        test_accuracy = []
        test_precision = []
        test_recall = []
        test_f1_score = []
        iterations = []
        acc_best_epoch  = 0
        for epoch in range(1, epochs + 1):
            loss = train(model, device, train_Loader, optimizer, epoch)

            iterations.append(epoch)
            loss_test, accuracy, precision, recall, F1_score = test(model, device, test_Loader)
            losses.append(loss_test)
            test_accuracy.append(accuracy)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1_score.append(F1_score)

            if acc_best_epoch < accuracy[-1]:
                acc_best_epoch = accuracy[-1]
                if acc_best_epoch >= 0.8:
                        torch.save(model.state_dict(),
                                   '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/pt_0.8/ckpt_' + str(
                                       trial) + '_' + str(
                                       acc_best_epoch) + '.pt')
        if acc_best_epoch > acc_best:
            acc_best = acc_best_epoch
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'losses_speaker.txt'), losses, fmt='%.6f')
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'accuracy_speaker.txt'), test_accuracy, fmt='%.6f')
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'precision_speaker.txt'), test_precision, fmt='%.6f')
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'recall_speaker.txt'), test_recall, fmt='%.6f')
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'f1_score_speaker.txt'), test_f1_score, fmt='%.6f')
        np.savetxt(os.path.join(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/raw_1Dconv_pytorch_speaker/result_train0.8',
            str(trial), 'iteration_speaker.txt'), iterations, fmt='%d')

        print("acc_best", acc_best)

