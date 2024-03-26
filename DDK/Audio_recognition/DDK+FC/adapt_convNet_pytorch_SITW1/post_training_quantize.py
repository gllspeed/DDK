import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import torchvision
import matplotlib.pyplot as plt
from model_sigmoid_quantize import Net
import numpy as np
import sklearn.metrics as sm

def direct_quantize(model, test_loader, device):

    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_forward(data)
        #if i % 500 == 0:
        #    break



def full_inference(model, test_loader,device):
    test_num = test_batch_size
    model.eval()
    test_loss = 0
    correct = 0

    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # print("test_data", data)
        # print("test_label", target)
        output, target = model(data, target)

        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / test_num
    ))
    accuracy = correct / test_num
    return accuracy


def quantize_inference(model, test_loader, device):
    test_num = test_batch_size
    model.eval()
    test_loss = 0
    correct = 0

    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # print("test_data", data)
        # print("test_label", target)
        output, target = model.quantize_inference(data, target)

        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        test_loss, 100. * correct / test_num
    ))
    accuracy = correct / test_num
    return accuracy



if __name__ == "__main__":
    num_class = 10

    momentum = 0.5
    save_model = False
    using_bn = False

    lr = 0.01
    train_batch_size = 1000
    test_batch_size = 250
    num_data = 1250
    num_every_class = 125
    ###########speaker###########
    input_mat = 'speaker_no_fft_norm_128_1250.mat'
    label_mat = 'targets_no_fft_norm_128_1250.mat'

    data_path = '/home/gaolili/deepLearning_project/Voiceprint_20201027/Input_file/speaker/10s/mat/DDK'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class,
                                          training='is_training')
    train_Loader = dataset.generate_dataLoader(train_data, shuffle=False, batch_size=train_batch_size)
    test_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class,
                                         training='is_testing')
    test_Loader = dataset.generate_dataLoader(test_data, shuffle=False, batch_size=test_batch_size)

    model = Net(num_class=num_class, num_every_class=num_every_class).to(device)
    model.load_state_dict(torch.load(
        '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/DDK_pt_0.8train_16/adapt_speaker_DDK_6_0.948.pt'))
    model.eval()


    full_inference(model, test_Loader, device)

    num_bits = 8
    ########初始化网络层的一些参数，如conv_module, qi, qw, qo这三个参数默认初始化为None。
    #############conv 会初始化输入输出的最大值，最小值，relu和maxpool2d统计输出的最大值，和最小值。
    model.quantize(num_bits=num_bits)
    model.eval()
    print('Quantization bit: %d' % num_bits)
    #前向跑一遍网络，更新每一层的min, max, scale, zero值。
    ###################比较耗时##################
    direct_quantize(model, train_Loader, device)

    ##############freeze 函数会在统计完 min、max 后对一些变量进行固化， tf的量化方式，在freeze的过程中，量化后的参数减零点，所以需在freeze之前通过调试代码来保存参数。
    model.freeze()
    ##########量化后的参数没有减零点，所以，可以在model.freeze（）后获取获取量化后模型的参数###############
    '''
    save_txt_path = '/home/gaolili/deepLearning_project/pytorch-quantization-demo_xinpian/save_weights/weights/'
    for name, parameters in model.named_parameters():
        print(name, parameters.size())
        parameter = parameters.cpu().detach().numpy().flatten()
        txt = os.path.join(save_txt_path, name.replace('.', '_')+'.txt')

        np.savetxt(txt, parameter,fmt='%d')
    '''
    #########################################
    quantize_inference(model, test_Loader, device)

    



    
