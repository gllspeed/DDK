import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
nltk.download('punkt')
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_file(file_path):
    """
    Read function for AG NEWS Dataset
    """
    data = pd.read_csv(file_path, names=["class", "title", "description"])
    texts = list(data['title'].values + ' ' + data['description'].values)
    texts = [word_tokenize(preprocess_text(sentence)) for sentence in texts]
    labels = [label-1 for label in list(data['class'].values)]  # label : 1~4  -> label : 0~3
    return texts, labels


def preprocess_text(string):
    """
    reference : https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def metrics(dataloader, losses, correct, y_hats, targets):
    avg_loss = losses / len(dataloader)
    accuracy = correct / len(dataloader.dataset) * 100
    precision = precision_score(targets, y_hats, average='macro')
    recall = recall_score(targets, y_hats, average='macro')
    f1 = f1_score(targets, y_hats, average='macro')
    cm = confusion_matrix(targets, y_hats)
    return avg_loss, accuracy, precision, recall, f1, cm

def alpha_beta_init_generate(mean, std):
    data = 0
    while(data<=0):
        data = np.random.normal(mean, std)
    return torch.Tensor([data]).to(device)

def alpha_beta_add_noise(m, alpha=True, beta=True):
    if alpha:
        alpha_mean = float(m.alpha.data)
        alpha_std = 0.01873 * 5.1
        alpha_noise = alpha_beta_init_generate(alpha_mean, alpha_std)
        m.alpha.data = alpha_noise
    if beta:
        beta_mean = float(m.beta.data)
        beta_std = 0.05411 * 5.1
        beta_noise = alpha_beta_init_generate(beta_mean, beta_std)
        m.beta.data = beta_noise

###########映射方案一###############
def calcScaleZeroPoint(min_val, max_val, map_min, map_max):

    scale = float((max_val - min_val) / (map_max - map_min))

    zero_point = map_max - max_val / scale

    if zero_point < map_min:
        zero_point = map_min
    elif zero_point > map_max:
        zero_point = map_max

    zero_point = int(zero_point)

    return scale, zero_point


def mapping(x, scale, zero_point, map_min, map_max):

    q_x = zero_point + x * scale
    q_x.clamp_(map_min, map_max)

    return q_x

def de_mapping(q_x, scale, zero_point):
    return (q_x - zero_point)/scale

def weight_map(weight, noise_mean, noise_std, map_min, map_max):
    m, n = weight.shape
    ##########weight map G################
    weight_abs_max = torch.abs(weight).max()
    zero_point = int((map_max + map_min) / 2)
    scale = (map_max - zero_point) / weight_abs_max

    weight_q = mapping(weight, scale, zero_point, map_min, map_max)
    weight_q_flatten = weight_q.flatten()

    noise = np.random.normal(noise_mean, noise_std, weight_q_flatten.shape[0])
    weight_q_flatten_noise = weight_q_flatten + torch.Tensor(noise).to(device)
    weight_q_flatten_noise.clamp_(map_min, map_max)

    weight_noise = de_mapping(weight_q_flatten_noise, scale, zero_point)
    return torch.reshape(weight_noise, (m,n))

def w_add_noise(module, noise_mean, noise_std, map_min, map_max):
    w = module.weight.data
    w.clamp_(-1, 1)
    w_noise = weight_map(w, noise_mean, noise_std, map_min, map_max)
    module.weight.data = w_noise