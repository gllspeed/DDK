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
        output, target = model(data, target)
        #print("train_label", target)
        #output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()
        #for name, parameters in model.named_parameters():
            #if name == 'MFL.alpha':
            #    parameters.data.clamp_(0.001, 1)
            #if name == 'MFL.beta':
            #    parameters.data.clamp_(0.001, 1)
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
        output,target = model(data, target)
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
    for trial in range(1,50):
        num_class = 10
        epochs = 500

        momentum = 0.5
        save_model = False
        using_bn = False

        input_data = 'MFL'  #########'MFL
        if input_data == 'MFL':
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

            data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/speaker/10s/mat/MFL'


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
            loss = train(model, device, train_Loader, optimizer, epoch, train_data_size)
            #print("alpha", model.MFL.alpha)
            #print("beta", model.MFL.beta)
            alphas.append(model.MFL.alpha.cpu().detach().numpy().tolist())
            betas.append(model.MFL.beta.cpu().detach().numpy().tolist())
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
                    if input_data == 'MFL':
                        torch.save(model.state_dict(),
                                   '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/model/adapt_speaker_MFL_' + str(trial)+'_'+str(
                                       acc_best_epoch) + '.pt')


        if acc_best_epoch > acc_best:
            acc_best = acc_best_epoch



        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'losses_speaker.txt'), losses, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'accuracy_speaker.txt'), test_accuracy, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'precision_speaker.txt'), test_precision, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'recall_speaker.txt'), test_recall, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'f1_score_speaker.txt'), test_f1_score, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'iteration_speaker.txt'), iterations, fmt='%d')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'alpha.txt'), alphas, fmt='%.6f')
        np.savetxt(os.path.join('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/result/MFL/speaker/quantize/result',str(trial), 'beta.txt'), betas, fmt='%.6f')

        print("acc_best", acc_best)


