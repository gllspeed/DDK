import torch
import torch.nn as nn
import torch.nn.functional as F
#################在单词的特征维度进行DDK操作################

class DDK_word_feature_dim(nn.Module):
    def __init__(self):
        super(DDK_word_feature_dim, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        ############DDK###########################

        nn.init.uniform_(self.alpha, 0.01, 1)
        nn.init.uniform_(self.beta, 0.01, 1)
        print("init_alpha", self.alpha)
        print("init_beta", self.beta)

    def matrix_norm(self, input):
        input_norm = (input - input.min()) / (input.max() - input.min())
        return input_norm

    def forward(self, input, args):
        l,m, n = input.shape
        input = self.matrix_norm(input)
        weight_matrix = torch.ones(l,m, n).to(args.device)
        for k in range(1, n):
            temp = 1 / weight_matrix[:,:, k - 1]
            weight_matrix[:,:, k] = weight_matrix[:,:, k - 1] + self.alpha * temp - self.beta * input[:,:, k - 1]
        return weight_matrix

##############在单词维度进行DDK操作####################
class DDK_word_dim(nn.Module):
    def __init__(self):
        super(DDK_word_dim, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        ############DDK###########################

        nn.init.uniform_(self.alpha, 0.01, 1)
        nn.init.uniform_(self.beta, 0.01, 1)
        print("init_alpha", self.alpha)
        print("init_beta", self.beta)

    def matrix_norm(self, input):
        input_norm = (input - input.min()) / (input.max() - input.min())
        return input_norm

    def forward(self, input, args):
        input = self.matrix_norm(input)
        input = input.transpose(1,2)
        l,m, n = input.shape
        weight_matrix = torch.ones(l,m, n).to(args.device)
        for k in range(1, n):
            temp = 1 / weight_matrix[:,:, k - 1]
            weight_matrix[:,:, k] = weight_matrix[:,:, k - 1] + self.alpha * temp - self.beta * input[:,:, k - 1]
        weight_matrix = weight_matrix.transpose(1,2)
        return weight_matrix

class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout, DDK_type):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(300, 128)
        self.fc = nn.Linear(128, class_num)
        self.sigmoid = nn.Sigmoid()
        self.DDK_type = DDK_type
        self.DDK_word_dim = DDK_word_dim()
        self.DDK_word_feature_dim = DDK_word_feature_dim()
    def forward(self, x, args):
        # x = |bs, seq_len|
        x_emb = self.embedding(x)
        if self.DDK_type == 'word_dim':
            output = self.DDK_word_dim(x_emb, args)
        elif self.DDK_type == 'word_feature_dim':
            output = self.DDK_word_feature_dim(x_emb, args)
        output = self.fc1(output)
        output = self.tanh(output).transpose(1, 2)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        output = self.fc(output)
        # output = |bs, class_num|
        return output