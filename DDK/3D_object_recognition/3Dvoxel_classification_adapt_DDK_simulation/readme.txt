仿真实验的模型如下（与软件算法模型一致）：
        MFL层的alpha和beta是多维的
        self.fc1 = nn.Linear(1024, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128,  bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, self.num_speakers_softmax, bias=False)

加载预训练模型，预训练模型存储在pretrained_model文件夹下;
加载预训练模型后，固定alpha, beta（相当于alpha, beta写入器件），含噪训练， 精调fc层的权重。精调后的权重存储在tuned_model文件夹下。
加载精调后的所有权重，在fc权重添加50次噪声，对测试集进行测试。
