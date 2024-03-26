import dataset
#from model_sigmoid_noquantization import Net, Net_quantize
from model_sigmoid_quantization import Net, Net_quantize
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

import torchvision

def quantize_aware_training(model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = lossLayer(output, target)
        #l2_alpha = 10
        #loss_alpha = model.add_loss(model, l2_alpha)
        #loss = loss + loss_alpha
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Quantize Aware Training Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

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



def full_inference(model, device, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.4f}%\n'.format(100. * correct / len(test_loader.dataset)))


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
    train_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class,
                                          training='is_training')
    train_Loader = dataset.generate_dataLoader(train_data, shuffle=False, batch_size=train_batch_size)
    test_data = dataset.load_voice_print(data_path, input_mat, label_mat, num_class, num_every_class,
                                         training='is_testing')
    test_Loader = dataset.generate_dataLoader(test_data, shuffle=False, batch_size=test_batch_size)


    if load_float_model:
        model_float = Net(num_class=num_class, num_every_class = num_every_class)
        #model.load_state_dict(torch.load('/home/gaolili/deepLearning_project/pytorch-quantization-demo_xinpian/ckpt/mnist_cnn_normalize(0,1)_2groupsConv.pt'))
        model_float.load_state_dict(torch.load('/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/MFL_pt_0.8train_16/adapt_speaker_MFL_6_0.948.pt'))

    #full_inference(model_float, device, test_Loader)

    for name, parameters in model_float.named_parameters():
        if 'MFL.alpha' in name:
            alpha =  parameters.cpu().detach().numpy()
        elif 'MFL.beta' in name:
            beta =  parameters.cpu().detach().numpy()
        elif 'fc1.weight' in name:
            fc1_weight = parameters.cpu().detach().numpy()
        elif 'fc.weight' in name:
            fc_weight = parameters.cpu().detach().numpy()
    Net_parameters = [fc1_weight, fc_weight]
    model = Net_quantize(alpha, beta, Net_parameters=Net_parameters, num_channels=1, num_bits = num_bits)
    model.to(device)

    optimizer  =  optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 55, 75], gamma=0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    ###########保存浮点模型权重的分布图###################
    ''' 
    for name, parameters in model.named_parameters():
        print(name, parameters.size())
        parameter = parameters.cpu().detach().numpy()
        from matplotlib import pyplot as plt
        n, bins, batches = plt.hist(parameter.ravel(), 128)
        plt.savefig('/home/gaolili/deepLearning_project/pytorch-quantization-demo_xinpian/plt_figure/'+name+'.png')
        plt.show()
    '''

    model.train()
    acc_max = 0
    for epoch in range(1, epochs + 1):
        #scheduler.step()
        quantize_aware_training(model, device, train_Loader, optimizer, epoch)
        test_acc = test(model, device, test_Loader)

        temp = test_acc
        #if epoch >=50 and test_acc>=acc_max:
        if test_acc > acc_max:
            acc_max = test_acc
            torch.save(model.state_dict(),
                      '/home/gaolili/deepLearning_project/Voiceprint_20201027/adapt_convNet_pytorch_SITW1/quantization_training/save_model/sigmoid_quantize/model_lsq_MFL_2bit_'+str(acc_max)+'.pt')
    #torch.save(model.state_dict(),
    #        '/home/gaolili/deepLearning_project/pytorch-quantization-demo_xinpian/ckpt/model_lsq_improved_alpha_weight_100.pt')
    #model.eval()
    #model.freeze()
    alpha_dict = {}
    for name, parameters in model.named_parameters():
        if 'alpha' in name:
            parameter = parameters.cpu().detach().numpy()
            alpha_dict[name] = torch.tensor(parameter[0])
            print(name, parameter)
    quantize_inference(model, test_Loader, device, alpha_dict, plot_confusion_matrix)






    
