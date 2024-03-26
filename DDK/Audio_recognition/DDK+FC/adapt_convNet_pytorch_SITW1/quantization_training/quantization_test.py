import dataset
#from from model_sigmoid_noquantization import Net, Net_quantize
from model_sigmoid_quantization import Net, Net_quantize
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

import torchvision


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)

        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))
    return correct / len(test_loader.dataset)



def quantize_inference(model, test_loader, device, alpha_dict, plot_confusion_matrix):
    ########10是类别数
    correct = 0
    class_total = [0] * 10
    true_class = [0] * 10
    conf_matrix = torch.zeros(10, 10)
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data, alpha_dict)
        pred = output.argmax(dim=1, keepdim=True)
        if compute_everyClass_acc:
            for index in range(len(target)):
                class_total[target[index]]+=1
                if pred[index] == target[index]:
                    true_class[pred[index]] += 1

        correct += pred.eq(target.view_as(pred)).sum().item()
    #print("pred", pred)

    if compute_everyClass_acc:
        class_acc = [true_class[index] / class_total[index] for index in range(10)]
        print("quantize_aware_acc", class_acc)
    print('\nTest set: Quant Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    num_bits = 2
    num_class = 10
    num_every_class = 125
    train_batch_size = 1000
    test_batch_size = 250
    seed = 1
    epochs = 100
    ##########高bit的学习率###############
    lr = 0.01
    #lr = 0.001
    #########4比特的学习率##############
    #lr = 0.0001
    momentum = 0.5
    save_model = True
    using_bn = False
    load_float_model = True
    plot_confusion_matrix = False
    compute_everyClass_acc = False
    root = "/home/gaolili/deepLearning_project/data/fashion-mnist/"
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_mat = 'speaker_no_fft_norm_128_1250.mat'
    label_mat = 'targets_no_fft_norm_128_1250.mat'

    data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/speaker/10s/mat/MFL'

    test_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class,
                                         training='is_testing')
    test_Loader = dataset.generate_dataLoader(test_data, shuffle=False, batch_size=test_batch_size)
    model = Net_quantize(num_channels=1, num_bits = num_bits)

    model.load_state_dict(torch.load(
        '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/quantization_training/save_model/sigmoid_quantize/model_lsq_MFL_2bit_0.748.pt'))
    model.to(device)
    ###########保存浮点模型权重的分布图###################

    alpha_dict = {}
    for name, parameters in model.named_parameters():
        if 'alpha' in name:
            parameter = parameters.cpu().detach().numpy()
            alpha_dict[name] = torch.tensor(parameter[0])
            print(name, parameter)

    quantize_inference(model, test_Loader, device, alpha_dict, plot_confusion_matrix)






    
