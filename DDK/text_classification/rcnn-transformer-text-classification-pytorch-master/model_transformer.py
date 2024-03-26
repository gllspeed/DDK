import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
class Positional_Encoding(nn.Module):
    '''
	params: embed-->word embedding dim      pad_size-->max_sequence_lenght
	Input: x
	Output: x + position_encoder
    '''
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    '''
	params: dim_model-->hidden dim      num_head
    '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape 回原来的形状
        out = self.fc(context)   # 全连接
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class ConfigTrans(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = args.class_num                   # 类别数
        self.num_epochs = 100                # epoch数
        self.batch_size = 128             # mini-batch大小
        self.pad_size = 64                     # 每句话处理成的长度(短填长切)，这个根据自己的数据集而定
        self.learning_rate = 0.001                    # 学习率
        self.embed = args.embedding_dim          # 字向量维度
        self.dim_model = args.embedding_dim      # 需要与embed一样
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5       # 多头注意力，注意需要整除
        self.num_encoder = 6    # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        config = ConfigTrans(args)
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        self.postion_embedding = Positional_Encoding(args.embedding_dim, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.copy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        out = self.fc1(out)
        return out
