import logging
import sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from constants import BATCH_SIZE_TEST, device
from conv_models import DeepSpeakerModel
import dataset
from utils import load_best_checkpoint, enable_deterministic
import os
import torch
from utils import read_mat
from utils import *
logger = logging.getLogger(__name__)


def softmax_test(dsm: DeepSpeakerModel, test_loader):
    test_num = BATCH_SIZE_TEST
    dsm.eval()
    test_loss = 0
    correct = 0######统计帧级别的准确率
    correct_sample = 0
    targets = []
    preds = []
    outputs = []
    class_total = [0] * 10
    true_class = [0] * 10
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, target = dsm(data, target)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        output_temp = output.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)
        outputs.extend(output_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()

    videos_num_everyclass = [0] * 10
    videos_correct_num_everyclass = [0] * 10
    ############概率投票法#####################
    '''
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        for j in range(num_frames):
            if temp_pred[j] == temp_target[j]:
                #index.append(i)
                correct_temp += 1
        if  correct_temp / num_frames > 0.3:
            correct_sample+=1
            videos_correct_num_everyclass[temp_target[0]]+=1
    '''
    ###########统计每类的总视频数和每类正确识别的视频数################



    #################hard voting#######################

    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        #print("temp_target",temp_target)


        pred_num = [0] * 10
        for j in range(num_frames):
            pred_num[int(temp_pred[j])]+=1
        #print("pred_num", pred_num)
        pred_label = pred_num.index(max(pred_num))
        #print("pred_label", pred_label)
        if pred_label == temp_target[0]:
            correct_sample += 1
            videos_correct_num_everyclass[temp_target[0]]+=1

    ##########soft voting#####################
    '''
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_output = np.array(outputs[i * num_frames:i * num_frames + num_frames][:])
        temp_output = np.mean(temp_output, axis=0)
        pred_label = np.argmax(temp_output)

        if pred_label == temp_target[0]:
            correct_sample += 1
            videos_correct_num_everyclass[temp_target[0]]+=1
    '''
    #print('\nTest set: Accuracy: {:.3f}%\n'.format(
    #    100. * correct_sample / (BATCH_SIZE_TEST / num_frames)
    #))

    test_loss /= test_num

    #############计算每一类的精度################
    accuracy = [videos_correct_num_everyclass[index] / videos_num_everyclass[index] for index in range(len(videos_num_everyclass))]

    accuracy.append(correct_sample / (BATCH_SIZE_TEST / num_frames))
    return accuracy

def softmax_test_addNoise(dsm: DeepSpeakerModel, test_loader, noise_mean = 0.01, noise_std = 0.01, map_min = 40, map_max = 300):
    test_num = BATCH_SIZE_TEST
    dsm.eval()
    test_loss = 0
    correct = 0######统计帧级别的准确率
    correct_sample = 0
    targets = []
    preds = []
    outputs = []
    class_total = [0] * 10
    true_class = [0] * 10
    '''
    for m in dsm.modules():
        if isinstance(m, nn.Linear):
            w_add_noise(m, noise_mean, noise_std, map_min, map_max)
    '''
    w_add_noise(dsm.fuse_fc1_and_bn1, noise_mean, noise_std, map_min, map_max)
    w_add_noise(dsm.fuse_fc2_and_bn2, noise_mean, noise_std, map_min, map_max)
    w_add_noise(dsm.fuse_fc3_and_bn3, noise_mean, noise_std, map_min, map_max)
    w_add_noise(dsm.fc4, noise_mean, noise_std, map_min, map_max)

    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, target = dsm(data, target)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)

        pred_temp = pred.squeeze(1).detach().cpu().numpy().tolist()
        output_temp = output.squeeze(1).detach().cpu().numpy().tolist()
        target_temp = target.detach().cpu().numpy().tolist()
        targets.extend(target_temp)
        preds.extend(pred_temp)
        outputs.extend(output_temp)

        correct += pred.eq(target.view_as(pred)).sum().item()

    videos_num_everyclass = [0] * 10
    videos_correct_num_everyclass = [0] * 10
    ############概率投票法#####################
    '''
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        for j in range(num_frames):
            if temp_pred[j] == temp_target[j]:
                #index.append(i)
                correct_temp += 1
        if  correct_temp / num_frames > 0.3:
            correct_sample+=1
            videos_correct_num_everyclass[temp_target[0]]+=1
    '''
    ###########统计每类的总视频数和每类正确识别的视频数################



    #################hard voting#######################

    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_pred = preds[i * num_frames:i * num_frames + num_frames]
        correct_temp = 0
        #print("temp_target",temp_target)


        pred_num = [0] * 10
        for j in range(num_frames):
            pred_num[int(temp_pred[j])]+=1
        #print("pred_num", pred_num)
        pred_label = pred_num.index(max(pred_num))
        #print("pred_label", pred_label)
        if pred_label == temp_target[0]:
            correct_sample += 1
            videos_correct_num_everyclass[temp_target[0]]+=1

    ##########soft voting#####################
    '''
    for i in range(int(len(targets) / num_frames)):

        temp_target = targets[i * num_frames:i * num_frames + num_frames]
        videos_num_everyclass[temp_target[0]]+=1
        temp_output = np.array(outputs[i * num_frames:i * num_frames + num_frames][:])
        temp_output = np.mean(temp_output, axis=0)
        pred_label = np.argmax(temp_output)

        if pred_label == temp_target[0]:
            correct_sample += 1
            videos_correct_num_everyclass[temp_target[0]]+=1
    '''
    #print('\nTest set: Accuracy: {:.3f}%\n'.format(
    #    100. * correct_sample / (BATCH_SIZE_TEST / num_frames)
    #))

    test_loss /= test_num

    #############计算每一类的精度################
    accuracy = [videos_correct_num_everyclass[index] / videos_num_everyclass[index] for index in range(len(videos_num_everyclass))]

    accuracy.append(correct_sample / (BATCH_SIZE_TEST / num_frames))
    return accuracy


'''
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_speakers_softmax = 10
    num_frames = 32
    is_DDK = True
    mat_path = '/home/gaolili/data/ModelNet10/ModelNet10_voxelized_mat'
    label_map = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7,
                 'table': 8, 'toilet': 9}
    _, _, kx_test, ky_test = read_mat(mat_path, label_map)
    num_samples = ky_test.shape[0]
    test_data = dataset.AudioDataset(kx_test, ky_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=num_samples, shuffle=False, num_workers=1,
                                              pin_memory=True)

    model = DeepSpeakerModel(num_channels=1, include_softmax=True, num_speakers_softmax=num_speakers_softmax, num_data=num_samples, is_DDK = is_DDK)
    dsm = model.to(device)
    #############每组实验，识别精度大于80%以上的模型的融合##########################
    accuracy_best = []
    for trial in ['1', '2', '3', '4', '5']:##########表示五组实验
        model_path = os.path.join('/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK/checkpoints/DDK/model_accuracy',trial, 'model')
        models_name = os.listdir(model_path)
        accuracy_all = []
        for i in range(len(models_name)):
            checkpoint = torch.load(os.path.join(model_path, models_name[i]))
            dsm.load_state_dict(checkpoint)

            accuracy = softmax_test(dsm, test_loader)
            accuracy_all.append(accuracy)
            #print(accuracy)
        accuracy_all = np.array(accuracy_all)
        accuracy_max = [max(accuracy_all[:,i]) for i in range(num_speakers_softmax)]
        #print("accuracy_max", accuracy_max)
        print("accuracy_max_mean", sum(accuracy_max)/num_speakers_softmax)
        accuracy_best.append(sum(accuracy_max)/num_speakers_softmax)

    accuracy_best_max = max(accuracy_best)
    accuracy_best_mean = np.mean(accuracy_best)
    accuracy_best_std = np.std(accuracy_best)
    print("accuracy_best_max", accuracy_best_max)
    print("accuracy_best_mean", accuracy_best_mean)
    print("accuracy_best_std", accuracy_best_std)
'''

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_speakers_softmax = 10
    num_frames = 32
    is_DDK = True
    mat_path = '/home/gaolili/data/ModelNet10/ModelNet10_voxelized_mat'
    label_map = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7,
                 'table': 8, 'toilet': 9}
    _, _, kx_test, ky_test = read_mat(mat_path, label_map)
    num_samples = ky_test.shape[0]
    test_data = dataset.AudioDataset(kx_test, ky_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=num_samples, shuffle=False, num_workers=1,
                                              pin_memory=True)

    model = DeepSpeakerModel(num_channels=1, include_softmax=True, num_speakers_softmax=num_speakers_softmax, num_data=num_samples, is_DDK = is_DDK)
    dsm = model.to(device)

    accuracy_all = []
    for trial in range(5):##########
        pretrained_state = torch.load(
            '/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK_simulation/checkpoints/DDK_noise_training/checkpoints-softmax_DDK/softmax_27_0.85.pt')

        model_dict = dsm.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        dsm.load_state_dict(model_dict)
        dsm.fuse()
        #########alpha, beta 添加噪声##############
        #alpha_beta_add_noise(dsm)
        accuracy = softmax_test_addNoise(dsm, test_loader, noise_mean = 0.20329, noise_std = 1.14726, map_min=40, map_max=300)
        accuracy_all.append(accuracy[-1])


    accuracy_all_max = max(accuracy_all)
    accuracy_all_mean = np.mean(accuracy_all)
    accuracy_all_std = np.std(accuracy_all)
    print("accuracy_best_max", accuracy_all_max)
    print("accuracy_best_mean", accuracy_all_mean)
    print("accuracy_best_std", accuracy_all_std)
