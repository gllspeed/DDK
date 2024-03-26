import logging
import os
import torch
import torch.optim as optim
import torch.nn as nn
import dataset
from tqdm import tqdm

from constants import BATCH_SIZE, CHECKPOINTS_SOFTMAX_DIR,  loss_accuracy_DIR,  learning_rate, momentum, resume_softmax, epochs, device
from constants import warm_up_epochs
from conv_models import VoxNet
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

from thop import clever_format
from thop import profile

def cal_flops(model):
    input = torch.randn(1, 1, 32, 32, 32).to(device)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    a = 0
def softmax_train(dsm: VoxNet, train_loader, epoch, optimizer):
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
def softmax_test(dsm: VoxNet, test_loader, epoch):
    dsm.eval()
    test_loss = 0
    correct = 0
    targets = []
    preds = []
    class_total = [0] * 10
    true_class = [0] * 10
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
    test_loss /= len(test_loader.dataset)

    #############计算每一类的精度################
    for index in range(len(targets)):
        class_total[targets[index]] += 1
        if preds[index] == targets[index]:
            true_class[preds[index]] += 1
    accuracy = [true_class[index] / class_total[index] for index in range(10)]
    accuracy_avg = correct / len(test_loader.dataset)
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))

    return accuracy, precision, recall, F1_score, test_loss


def start_training(kx_train, ky_train, kx_test, ky_test, trial=1):

    ensures_dir(CHECKPOINTS_SOFTMAX_DIR)
    ##########将best_acc写到txt里边
    #file = open('/home/gaolili/deepLearning_project/deep-speaker_sitw_smallCNN_pytorch/pre-training/result/acc.txt', 'a+')
    logger.info('Softmax pre-training.')
    num_speakers_softmax = 10
    train_data = dataset.AudioDataset(kx_train, ky_train)
    test_data = dataset.AudioDataset(kx_test, ky_test)
    model = VoxNet(num_speakers_softmax=num_speakers_softmax)
    dsm = model.to(device)
    cal_flops(dsm)
    #optimizer = optim.Adam(dsm.parameters(), lr=learning_rate, weight_decay = 0.001)
    optimizer = optim.Adam(dsm.parameters(), lr=learning_rate, weight_decay=0.01)
    #optimizer = optim.SGD(dsm.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.001)
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

    if resume_softmax:
        if os.path.isfile(resume_softmax):
            print('=> loading checkpoint {}'.format(resume_softmax))
            checkpoint = torch.load(resume_softmax)
            dsm.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            initial_epoch = checkpoint['epoch']
        else:
            print("=> no found checkpoint {}".format(resume_softmax))



    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                                              pin_memory=True)
    early_stopping_flag = False
    acc_best = 0
    for epoch in range(initial_epoch, epochs):
        scheduler.step()
        acc_train, loss_train = softmax_train(dsm, train_loader, epoch, optimizer)
        if acc_train is not None and loss_train is not None:
            train_accuracy.append(acc_train)
            train_losses.append(loss_train)
        if epoch % 10 == 0:
            #torch.save(dsm.state_dict(),os.path.join(CHECKPOINTS_SOFTMAX_DIR,'softmax_'+str(epoch)+'.pt'))
            accuracy, precision, recall, F1_score, loss_test = softmax_test(dsm, test_loader, epoch)
            test_accuracy.append(accuracy)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1_score.append(F1_score)
            test_losses.append(loss_test)
            if accuracy[-1] > acc_best:
                acc_best = accuracy[-1]
                if accuracy[-1] > 0.8:
                    torch.save(dsm.state_dict(), os.path.join(CHECKPOINTS_SOFTMAX_DIR, 'softmax_' + str(accuracy[-1]) + '.pt'))
            #if early_stopping_flag:
            #    print("acc_best", str(acc_best))
            #    break

    np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),'test_acc_'+str(acc_best)+'.txt'),test_accuracy, fmt='%.6f')
    np.savetxt(os.path.join(loss_accuracy_DIR, str(trial), 'test_precision_' + str(acc_best) + '.txt'),test_precision, fmt='%.6f')
    np.savetxt(os.path.join(loss_accuracy_DIR, str(trial), 'test_recall_' + str(acc_best) + '.txt'),test_recall, fmt='%.6f')
    np.savetxt(os.path.join(loss_accuracy_DIR, str(trial), 'test_f1_' + str(acc_best) + '.txt'),test_f1_score, fmt='%.6f')
    np.savetxt(os.path.join(loss_accuracy_DIR, str(trial),'test_loss_'+ str(acc_best)+ '.txt'),test_losses, fmt='%.6f')




