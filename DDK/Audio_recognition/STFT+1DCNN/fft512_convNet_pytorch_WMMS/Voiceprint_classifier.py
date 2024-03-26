"""
    建议：安装cupy运行该案例
"""
import os
os.environ.setdefault('GPU', "True")

from Light import nn, light
from Light import optim
from Light.utils import dataset
from Light.utils import train_one_epoch, evaluate
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        self.Hidden = nn.Linear(8, 4, bias = True)
        self.activate = nn.Sigmoid()
        self.fc = nn.Linear(4, 4, bias = True)

    def forward(self, inputs):
        x = self.Hidden(inputs)
        x = self.activate(x)
        output = self.fc(x)
        return output


if __name__ == '__main__':
    data_path = '/home/gaolili/deepLearning_project/Lightv3/Light/Dataset/Voiceprint/WhaleSong_device_20200819_RESET.mat'
    batch_size = 32
    train_data = dataset.load_voice_print(data_path, training='is_training')
    val_data = dataset.load_voice_print(data_path, training='is_val')
    test_data = dataset.load_voice_print(data_path, training='is_test')
    train_Loader = dataset.generate_dataLoader(train_data, batch_size = batch_size)
    val_Loader = dataset.generate_dataLoader(val_data, batch_size = batch_size)
    test_Loader = dataset.generate_dataLoader(test_data, batch_size = batch_size)
    model = MlpNet()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, centered=True, momentum=0.6)
    # loss_fn = light.binary_cross_entropy_with_logits_one_hot
    loss_fn = light.softmax_cross_entropy_with_logits
    #evaluate(model, val_dataloader)
    for epoch in range(200):
        train_one_epoch(model, optimizer, loss_fn, train_Loader, epoch)
        #scheduler.step()
        evaluate(model, val_Loader)
    evaluate(model, test_Loader)



