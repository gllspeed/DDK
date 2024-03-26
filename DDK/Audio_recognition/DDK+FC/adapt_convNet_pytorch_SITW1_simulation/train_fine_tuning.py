import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import sklearn.metrics as sm
from thop import clever_format
from thop import profile
from utils import *
from model_fine_tuning import *
def train(model, device, train_loader, optimizer, epoch, train_batch_size, noise_mean = None, noise_std = None, map_min = None, map_max = None):
    train_num = train_batch_size
    model.train()
    '''
    for m in model.modules():
        if isinstance(m, DDK):
            if epoch == 1:
                alpha_beta_add_noise(m)
    '''
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
        #for name, parameters in model.named_parameters():
            #if name == 'DDK.alpha':
            #    parameters.data.clamp_(0.001, 1)
            #if name == 'DDK.beta':
            #    parameters.data.clamp_(0.001, 1)
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
    print('\nTrain set: Accuracy: {:.6f}%\n'.format(100. * correct / len(train_loader.dataset)))
    avg_loss = total_loss/len(train_loader.dataset)
    return avg_loss

def test(model, device, test_loader, test_batch_size, num_class, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300):
    #test_num = test_batch_size
    model.eval()
    test_loss = 0
    correct = 0

    ############一个epoch, 参数添加噪声, alpha, beta写进去已固定， test添加噪声测试完后，之后训练时权重已经添加了噪声，相当于在训练集添加了噪声####################
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

    test_loss /= len(test_loader.dataset)
    #############计算每一类的精度################
    for index in range(len(targets)):
        class_total[targets[index]] += 1
        if preds[index] == targets[index]:
            true_class[preds[index]] += 1
    accuracy = [true_class[index] / class_total[index] for index in range(num_class)]
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
    return test_loss, accuracy, precision, recall, F1_score

def cal_flops(model):
    input = torch.randn(1, 128).to(device)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    a = 0
if __name__ == "__main__":
    for trial in range(1,51):
        num_class = 10
        epochs = 500

        momentum = 0.5
        save_model = False
        using_bn = False

        input_data = 'DDK'  #########'DDK
        if input_data == 'DDK':
            lr = 0.01
            train_batch_size = 1000#875#1000#800
            test_batch_size = 250#188#250#200
            train_data_size = 1000#875#1000#800
            train_num_everyclass = 100#80
            test_num_everyclass = 25#20
            test_data_size = 250#188#250#200
            num_data = 1250#1000
            num_every_class = 125#100
            ###########speaker###########
            input_mat = 'speaker_no_fft_norm_128_1250.mat'
            label_mat = 'targets_no_fft_norm_128_1250.mat'

            data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/speaker/10s/mat/DDK'


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ###########################训练audio##################
        acc_best = 0


        train_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class, training='is_training')
        train_Loader = dataset.generate_dataLoader(train_data, shuffle = False, batch_size=train_batch_size)
        test_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class, training='is_testing')
        test_Loader = dataset.generate_dataLoader(test_data, shuffle=False, batch_size=test_batch_size)
        '''
        val_data = dataset.load_voice_print(data_path, training='is_val', sed=seed)
        test_data = dataset.load_voice_print(data_path, training='is_test', sed=seed)
        
        val_Loader = dataset.generate_dataLoader(val_data, batch_size=batch_size)
        test_Loader = dataset.generate_dataLoader(test_data, batch_size=batch_size)
        '''
        model = Net(num_class=num_class, num_every_class = num_every_class).to(device)
        model.load_state_dict(torch.load(
            '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/pretrained_model/adapt_speaker_DDK_6_0.948.pt'))
        #cal_flops(model)
        ############alpha, beta的值添加噪声###########
        for m in model.modules():
            if isinstance(m, DDK):
                alpha_beta_add_noise(m)

        optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=0.001)
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
            loss = train(model, device, train_Loader, optimizer, epoch, train_data_size, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300)
            #print("alpha", model.DDK.alpha)
            #print("beta", model.DDK.beta)
            alphas.append(model.DDK.alpha.cpu().detach().numpy().tolist())
            betas.append(model.DDK.beta.cpu().detach().numpy().tolist())
            iterations.append(epoch)
            if epoch % 10 ==0:
                loss_test, accuracy, precision, recall, F1_score = test(model, device, test_Loader, test_data_size, num_class, noise_mean = 0.20329, noise_std = 1.14726, map_min = 40, map_max = 300)
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
                                       '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/DDK_pt_0.8train_16_fine_tuning/adapt_speaker_DDK_' + str(trial)+'_'+str(
                                           acc_best_epoch) + '.pt')


        if acc_best_epoch > acc_best:
            acc_best = acc_best_epoch



        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'losses_speaker.txt'), losses, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'accuracy_speaker.txt'), test_accuracy, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'precision_speaker.txt'), test_precision, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'recall_speaker.txt'), test_recall, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'f1_score_speaker.txt'), test_f1_score, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'iteration_speaker.txt'), iterations, fmt='%d')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'alpha.txt'), alphas, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1_simulation/result/DDK/speaker/0.8_train_16_fine_tuning',str(trial), 'beta.txt'), betas, fmt='%.6f')

        print("acc_best", acc_best)


