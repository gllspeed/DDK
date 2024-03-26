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

def cal_flops(model):
    input = torch.randn(1, 64).to(device)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    a = 0
def train(model, device, train_loader, optimizer, epoch, train_batch_size):
    train_num = train_batch_size
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    total_loss = 0
    count = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        count +=1
        data, target = data.to(device), target.to(device)
        #print("train_data", data)
        #print("train_label", target)
        optimizer.zero_grad()
        output = model(data)
        #print("train_label", target)
        #output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()
        for name, parameters in model.named_parameters():
            if name == 'DDK.alpha':
                parameters.data.clamp_(0.001, 1)
            if name == 'DDK.beta':
                #parameters.data.clamp_(0.3, 2)
                parameters.data.clamp_(0.001, 1)
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
    print('\nTrain set: Accuracy: {:.6f}%\n'.format(100. * correct / train_num))
    avg_loss = total_loss/train_num
    return avg_loss

def test(model, device, test_loader, test_batch_size, num_class):
    test_num = test_batch_size

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
        #print("test_data", data)
        #print("test_label", target)
        output = model(data)
        #print("test_label", target)
        #output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_num
    #############计算每一类的精度################
    for index in range(len(targets)):
        class_total[targets[index]] += 1
        if preds[index] == targets[index]:
            true_class[preds[index]] += 1
    accuracy = [true_class[index] / class_total[index] for index in range(num_class)]
    accuracy_avg = correct / test_num
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
        test_loss, 100. * correct / test_num
    ))
    return test_loss, accuracy, precision, recall, F1_score

if __name__ == "__main__":
    for trial in range(1,51):
        num_class = 4
        epochs = 500
        lr = 0.01
        momentum = 0.5
        save_model = False
        using_bn = False

        input_data = 'DDK'  #########'DDK
        if input_data == 'DDK':
            train_batch_size = 480#480#420#320  # 所有样本数
            test_batch_size = 120#120#90#80  # 所有样本数
            train_data_size = 480#480#420#320
            train_num_everyclass = 90#80
            test_num_everyclass = 20
            test_data_size = 120#120#90#80
            num_data = 600#400
            num_every_class = 150#100

            ###########whale###############
            #input_mat = 'WhaleSong_no_fft_abs_mean_32.mat'
            input_mat = 'WhaleSong_no_fft_abs_mean_64.mat'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ###########################训练audio##################
        acc_best = 0

        seed = 10


        data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file'
        train_data = dataset.load_voice_print(data_path, input_mat, num_class, num_every_class, training='is_training', sed=seed)
        train_Loader = dataset.generate_dataLoader(train_data, batch_size=train_batch_size)
        test_data = dataset.load_voice_print(data_path, input_mat, num_class, num_every_class, training='is_testing',sed=seed)
        test_Loader = dataset.generate_dataLoader(test_data, batch_size=test_batch_size)

        model = Net(num_class=num_class, num_every_class = num_every_class).to(device)
        cal_flops(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        losses = []
        accs = []
        test_accuracy = []
        test_precision = []
        test_recall = []
        test_f1_score = []
        alphas = []
        betas = []
        iterations = []
        acc_best_epoch  = 0
        for epoch in range(1, epochs + 1):
            loss = train(model, device, train_Loader, optimizer, epoch, train_data_size)
            print("alpha", model.DDK.alpha)
            print("beta", model.DDK.beta)
            alphas.append(model.DDK.alpha.cpu().detach().numpy()[0])
            betas.append(model.DDK.beta.cpu().detach().numpy()[0])
            iterations.append(epoch)
            loss_test, accuracy, precision, recall, F1_score = test(model, device, test_Loader, test_data_size, num_class)
            losses.append(loss_test)
            test_accuracy.append(accuracy)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1_score.append(F1_score)

            if acc_best_epoch < accuracy[-1]:
                acc_best_epoch = accuracy[-1]
                if acc_best_epoch>=0.9:
                    if input_data == 'DDK':
                        torch.save(model.state_dict(),
                                   '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/pt_64_0.8_8/adapt_speaker_WMMS_' + str(trial)+'_'+ str(
                                       acc_best_epoch) + '.pt')

        if acc_best_epoch > acc_best:
            acc_best = acc_best_epoch



        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'losses_speaker.txt'), losses, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'accuracy_speaker.txt'), test_accuracy, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'precision_speaker.txt'), test_precision, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'recall_speaker.txt'), test_recall, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'f1_score_speaker.txt'), test_f1_score, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'iteration_speaker.txt'), iterations, fmt='%d')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'alpha.txt'), alphas, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_WMMS/result/DDK/whale_64_0.8train_8',str(trial), 'beta.txt'), betas, fmt='%.6f')

        print("acc_best", acc_best)


