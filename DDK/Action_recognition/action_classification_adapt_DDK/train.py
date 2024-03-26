import logging
import os
import torch
import torch.optim as optim
import torch.nn as nn
from pytorchtools import EarlyStopping
import dataset
from tqdm import tqdm

from constants import BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, CHECKPOINTS_SOFTMAX_DIR,  loss_accuracy_DIR,  learning_rate, momentum, resume_softmax, epochs, device
from constants import warm_up_epochs, num_frames
from conv_models import DeepSpeakerModel
from utils import load_best_checkpoint, ensures_dir
import numpy as np
import math
from utils import WarmUpLR
import sklearn.metrics as ms
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Otherwise it's just too much logging from Tensorflow...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def softmax_train(dsm: DeepSpeakerModel, train_loader, epoch, optimizer):
    train_num = BATCH_SIZE_TRAIN
    dsm.train()
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = dsm(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = lossLayer(output, target)
        total_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(dsm.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))
    avg_loss = total_loss / len(train_loader.dataset)
    if epoch%10==0:
        print('Accuracy: {:.3f}%\n'.format(
            100. * correct / len(train_loader.dataset)
        ))
        return correct / len(train_loader.dataset), avg_loss
    else:
        return None, None


def softmax_test(dsm: DeepSpeakerModel, test_loader, epoch, early_stopping, num_class, early_stopping_flag = False):
    test_num = BATCH_SIZE_TEST
    dsm.eval()
    test_loss = 0
    correct = 0######统计帧级别的准确率
    correct_sample = 0
    targets = []
    preds = []
    class_total = [0] * num_class
    true_class = [0] * num_class
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = dsm(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()


    '''
    ###############概率投票法#####################
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        for j in range(num_frames):
            if temp_pred[j] == temp_target[j]:
                index.append(i)
                correct_temp += 1
        if  correct_temp / num_frames > 0.5:
            correct_sample+=1
    '''
    ###########硬投票法###统计每类的总视频数和每类正确识别的视频数################
    videos_num_everyclass = [0] * num_class
    videos_correct_num_everyclass = [0] * num_class
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        #print("temp_target",temp_target)


        pred_num = [0] * num_class
        for j in range(num_frames):
            pred_num[int(temp_pred[j])]+=1
        #print("pred_num", pred_num)
        pred_label = pred_num.index(max(pred_num))
        #print("pred_label", pred_label)
        if pred_label == temp_target[0]:
            correct_sample += 1
            videos_correct_num_everyclass[temp_target[0]]+=1
    print('\nTest set: Accuracy: {:.3f}%\n'.format(
        100. * correct_sample / (BATCH_SIZE_TEST / num_frames)
    ))

    test_loss /= test_num

    #############计算每一类的精度################
    accuracy = [videos_correct_num_everyclass[index] / videos_num_everyclass[index] for index in range(len(videos_num_everyclass))]

    accuracy_avg = correct_sample / (BATCH_SIZE_TEST / num_frames)
    accuracy.append(accuracy_avg)



    precision = ms.precision_score(targets, preds, average=None).tolist()
    precision_avg = np.mean(precision)
    precision.append(precision_avg)

    #########macro###############
    recall = ms.recall_score(targets, preds, average=None).tolist()
    recall_avg = np.mean(recall)
    recall.append(recall_avg)

    F1_score = ms.f1_score(targets, preds, average=None).tolist()
    F1_score_avg = np.mean(F1_score)
    F1_score.append(F1_score_avg)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
    #    test_loss, 100. * correct / test_num
    #))
    early_stopping(test_loss, dsm)
    if early_stopping.early_stop:
        print("Early stopping")
        early_stopping_flag = True
    return early_stopping_flag, accuracy, precision, recall, F1_score, test_loss


def start_training(kx_train, ky_train, kx_test, ky_test,  random_seed=1,  trial=1, is_DDK = True):

    ensures_dir(CHECKPOINTS_SOFTMAX_DIR)
    ##########将best_acc写到txt里边
    #file = open('/home/gaolili/deepLearning_project/deep-speaker_sitw_smallCNN_pytorch/pre-training/result/acc.txt', 'a+')
    early_stopping = EarlyStopping(patience=20, verbose=True)

    logger.info('Softmax pre-training.')
    num_speakers_softmax = 10
    train_data = dataset.AudioDataset(kx_train, ky_train)
    test_data = dataset.AudioDataset(kx_test, ky_test)
    model = DeepSpeakerModel(num_channels=1, include_softmax=True, num_speakers_softmax=num_speakers_softmax, is_DDK=is_DDK)
    dsm = model.to(device)
    optimizer = optim.Adam(dsm.parameters(), lr=learning_rate, weight_decay = 0.001)
    #optimizer = optim.Adam(dsm.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(dsm.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200, 300], gamma=0.1)
    #iter_per_epoch = len(train_data) / BATCH_SIZE
    #warmup_scheduler =  WarmUpLR(optimizer, iter_per_epoch * warm_up_epochs)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 150 - warmup_scheduler)
    #warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
    #    else 0.5 * (math.cos((epoch - warm_up_epochs) / (max_num_epochs - warm_up_epochs) * math.pi) + 1)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    initial_epoch = 0
    train_accuracy = []
    train_losses = []
    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1_score = []
    test_losses = []
    if is_DDK:
        alphas = []
        betas = []
    if resume_softmax:
        if os.path.isfile(resume_softmax):
            print('=> loading checkpoint {}'.format(resume_softmax))
            checkpoint = torch.load(resume_softmax)
            dsm.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            initial_epoch = checkpoint['epoch']
        else:
            print("=> no found checkpoint {}".format(resume_softmax))



    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=1,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=1,
                                              pin_memory=True)
    early_stopping_flag = False
    acc_best = 0
    for epoch in range(initial_epoch, epochs):
        scheduler.step()
        acc_train, loss_train = softmax_train(dsm, train_loader, epoch, optimizer)
        if is_DDK:
            alphas.append(dsm.alpha.cpu().detach().numpy().tolist())
            betas.append(dsm.beta.cpu().detach().numpy().tolist())
            #print("alpha", dsm.alpha)
            #print("beta", dsm.beta)
        if acc_train is not None and loss_train is not None:
            train_accuracy.append(acc_train)
            train_losses.append(loss_train)
        if epoch % 10 == 0:
            #torch.save(dsm.state_dict(),os.path.join(CHECKPOINTS_SOFTMAX_DIR,'softmax_'+str(epoch)+'.pt'))
            early_stopping_flag, accuracy, precision, recall, F1_score, loss_test = softmax_test(dsm, test_loader, epoch, early_stopping, num_speakers_softmax, early_stopping_flag)
            test_accuracy.append(accuracy)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1_score.append(F1_score)
            test_losses.append(loss_test)
            if accuracy[-1] > acc_best:
                acc_best = accuracy[-1]
                #if accuracy[-1] > 0.68:
                torch.save(dsm.state_dict(), os.path.join(CHECKPOINTS_SOFTMAX_DIR, 'softmax_' + str(trial)+'_'+ str(accuracy[-1]) + '.pt'))
            #if early_stopping_flag:
            #    print("acc_best", str(acc_best))
            #    break
        if epoch == 150 or epoch == 200 or epoch ==249 or epoch == 499:
            np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),
                                    'test_acc_DDK_' + str(acc_best) + '_' + str(epoch) + '.txt'), test_accuracy,
                       fmt='%.6f')

            np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),
                                    'test_precision_DDK_' + str(acc_best) + '_' + str(epoch) + '.txt'), test_precision,
                       fmt='%.6f')
            np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),
                                    'test_recall_DDK_' + str(acc_best) + '_' + str(epoch) + '.txt'), test_recall,
                       fmt='%.6f')
            np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),
                                    'test_f1_score_DDK_' + str(acc_best) + '_' + str(epoch) + '.txt'), test_f1_score,
                       fmt='%.6f')
            np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),
                                    'test_loss_DDK_' + str(acc_best) + '_' + str(epoch) + '.txt'), test_losses,
                       fmt='%.6f')

            if is_DDK:
                np.savetxt(os.path.join(loss_accuracy_DIR, str(trial), str(epoch) + '_' + 'alpha.txt'), alphas,
                           fmt='%.6f')
                np.savetxt(os.path.join(loss_accuracy_DIR, str(trial), str(epoch) + '_' + 'beta.txt'), betas,
                           fmt='%.6f')




