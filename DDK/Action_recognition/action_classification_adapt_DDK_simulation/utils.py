import logging
import os
import random
import shutil
from glob import glob
import cv2
import click
import dill
import numpy as np
import pandas as pd
from natsort import natsorted
from collections import defaultdict
import shutil
from constants import TRAIN_TEST_RATIO
import torch
from torch.optim.lr_scheduler import _LRScheduler
logger = logging.getLogger(__name__)


def find_files(directory, ext='wav'):
    return sorted(glob(directory + f'/**/*.{ext}', recursive=True))


def init_pandas():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


def create_new_empty_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def ensure_dir_for_filename(filename: str):
    ensures_dir(os.path.dirname(filename))


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


class ClickType:

    @staticmethod
    def input_file(writable=False):
        return click.Path(exists=True, file_okay=True, dir_okay=False,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def input_dir(writable=False):
        return click.Path(exists=True, file_okay=False, dir_okay=True,
                          writable=writable, readable=True, resolve_path=True)

    @staticmethod
    def output_file():
        return click.Path(exists=False, file_okay=True, dir_okay=False,
                          writable=True, readable=True, resolve_path=True)

    @staticmethod
    def output_dir():
        return click.Path(exists=False, file_okay=False, dir_okay=True,
                          writable=True, readable=True, resolve_path=True)


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned


def load_best_checkpoint(checkpoint_dir):
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    if len(checkpoints) != 0:
        return checkpoints[-1]
    return None


def delete_older_checkpoints(checkpoint_dir, max_to_keep=5):
    assert max_to_keep > 0
    checkpoints = natsorted(glob(os.path.join(checkpoint_dir, '*.h5')))
    checkpoints_to_keep = checkpoints[-max_to_keep:]
    for checkpoint in checkpoints:
        if checkpoint not in checkpoints_to_keep:
            os.remove(checkpoint)


def enable_deterministic():
    print('Deterministic mode enabled.')
    np.random.seed(123)
    random.seed(123)


def load_pickle(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading PKL file: {file}.')
    with open(file, 'rb') as r:
        return dill.load(r)


def load_npy(file):
    if not os.path.exists(file):
        return None
    logger.info(f'Loading NPY file: {file}.')
    return np.load(file)


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
def matrix_norm(input):
    input_norm = input / 255.
    return input_norm
#############按每个视频的帧数做DDK操作########################
'''
def DDK(input, a, b, num_frames=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.transpose(input, 0, 1).to(device)
    input = matrix_norm(input)
    m, n,  = input.shape
    weight_matrix = torch.ones(m, n).to(device)
    num_videos = int(input.shape[1] / num_frames)
    pad = [num_frames] * num_videos
    start_index = 0
    for i in range(1, len(pad) + 1):
        for k in range(start_index, start_index + pad[i - 1]):
            if k != start_index:
                temp = 1 / weight_matrix[:, k - 1]
                weight_matrix[:, k] = weight_matrix[:, k - 1] + a * temp - b * input[:, k - 1]
        start_index += pad[i - 1]

    weight_matrix = torch.transpose(weight_matrix, 0, 1)
    return weight_matrix
'''

'''
#############按每个视频的帧数做DDK操作(并行)########################
def DDK(input, a, b, num_frames=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n, m = input.shape
    num_videos = int(n / num_frames)
    input = torch.reshape(torch.transpose(input, 0, 1), (m, num_videos, num_frames)).to(device)
    input = matrix_norm(input)
    weight_matrix = torch.ones(m, num_videos,num_frames).to(device)
    for k in range(1, num_frames):
        temp = 1 / weight_matrix[:, :, k - 1]
        weight_matrix[:, :, k] = weight_matrix[:, :, k - 1] + a[k] * temp - b[k] * input[:, :, k - 1]
    weight_matrix = torch.reshape(weight_matrix, (m, n))
    weight_matrix = torch.transpose(weight_matrix, 0, 1)
    return weight_matrix
'''



###########按每个视频的特征维度做DDK操作########################

def DDK(input, a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.reshape(input, (-1,80,60))
    l,n, m = input.shape
    input = matrix_norm(input)
    weight_matrix = torch.ones(l, n, m).to(device)
    for k in range(1, m):
        temp = 1 / weight_matrix[:,:, k - 1]
        #weight_matrix[:,:, k] = weight_matrix[:,:, k - 1] + a[k] * temp - b[k] * input[:,:, k - 1]
        weight_matrix[:, :, k] = weight_matrix[:, :, k - 1] + a * temp - b * input[:, :, k - 1]
    weight_matrix = torch.flatten(weight_matrix, start_dim=1)
    return weight_matrix

'''
def DDK(input, a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.transpose(input, 0, 1).to(device)
    input = matrix_norm(input)
    m, n,  = input.shape
    weight_matrix = torch.ones(m, n).to(device)

    for k in range(1, n):
        temp = 1 / weight_matrix[:,k-1]
        weight_matrix[:, k] = weight_matrix[:, k-1] + a * temp - b * input[:, k-1]

    weight_matrix = torch.transpose(weight_matrix, 0, 1)
    return weight_matrix
'''
###########生成数据集#########################
def compute_MinVideoFrame(data_root):
    num_frames_min = float('inf')
    num_frames_max = 0
    for root, dirs, _ in os.walk(data_root):
        for dir in dirs:
            video_path = os.path.join(root, dir)
            for video_name in os.listdir(video_path):
                video = cv2.VideoCapture(os.path.join(video_path, video_name))
                num_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_frame < num_frames_min:
                    num_frames_min = num_frame
                if num_frame > num_frames_max:
                    num_frames_max = num_frame
    print("视频最小帧数{}".format(num_frames_min))
    print("视频最大帧数{}".format(num_frames_max))
    return num_frames_min, num_frames_max

def read_video(video_path, num_frames):

    video = cv2.VideoCapture(video_path)
    #video.open(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print('视频中的图像宽度{}'.format(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    #print('视频中的图像高度{}'.format(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #print('视频帧率{}'.format(video.get(cv2.CAP_PROP_FPS)))
    #print('视频帧数{}'.format(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("video_frames",video_frames)
    print("video_path")
    id = 0

    resize_divisor = 4
    images = np.zeros((num_frames, int(width/resize_divisor) * int(height/resize_divisor)))

    if video_frames <= num_frames:

        while id < video_frames:
            success, frame = video.read()
            if success == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(frame_gray, (int(width / resize_divisor), int(height / resize_divisor)), interpolation=cv2.INTER_CUBIC)
                image_flatten = np.reshape(image, (int(width / resize_divisor) * int(height / resize_divisor)))
                images[id, :] = image_flatten
            id += 1
    else:
        is_read = 0
        pad = int(video_frames / num_frames)

        while id < video_frames:
            #print("id", id)
            #print("is_read", is_read)

            if not id % pad and is_read < num_frames:
                video.set(cv2.CAP_PROP_POS_FRAMES, id)
                success, frame = video.read()
                #print(id)
                if success == True:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    image = cv2.resize(frame_gray, (int(width / resize_divisor), int(height / resize_divisor)), interpolation=cv2.INTER_CUBIC)
                    image_flatten = np.reshape(image, (int(width / resize_divisor) * int(height / resize_divisor)))
                    images[is_read, :] = image_flatten
                    is_read += 1

                else:
                    j = id + 1
                    video.set(cv2.CAP_PROP_POS_FRAMES, j)
                    success, frame = video.read()
                    while(success == False and j < video_frames):
                        j += 1
                        video.set(cv2.CAP_PROP_POS_FRAMES, j)
                        success, frame = video.read()

                    if success:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        image = cv2.resize(frame_gray, (int(width / resize_divisor), int(height / resize_divisor)), interpolation=cv2.INTER_CUBIC)
                        image_flatten = np.reshape(image, (int(width / resize_divisor) * int(height / resize_divisor)))
                        images[is_read, :] = image_flatten
                        is_read += 1
                    id = j

            id += 1

    return images
def remove_nullFrames(video_path):
    images = []
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video.open(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_index in range(video_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video.read()
        if success:
            frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
            images.append(frame.tolist())
    return images, width, height
def read_video_RGB(video_path, num_frames):
    images, width, height  = remove_nullFrames(video_path)
    video_frames = len(images)
    resize_divisor = 4
    images_num_frames = np.zeros((num_frames, 112, 112, 3))

    images_array = np.array(images)

    if video_frames <= num_frames:

        #images_flatten = np.reshape(images_array, (-1, (int(width / resize_divisor) * int(height / resize_divisor) * 3)))
        images_num_frames[0:video_frames, :, :, :] = images_array
    else:
        #images_flatten = np.reshape(images_array, (-1,(int(width / resize_divisor) * int(height / resize_divisor) * 3)))
        pad = int(video_frames / num_frames)
        for i in range(num_frames):
            images_num_frames[i, :, :, :] = images_array[i*pad, :, :, :]

    return images_num_frames

def generate_pytorch_data_shuffle(data_root, label_map, is_training=True):
    kx = []
    ky = []
    num_frames = 16

    for root, dirs, _ in os.walk(data_root):
        for dir in dirs:
            print("dir{}".format(dir))
            label = label_map[dir]
            kx_temp = []
            video_path = os.path.join(root, dir)
            for video_name in os.listdir(video_path):
                print("video_name{}".format(video_name))
                image = read_video(os.path.join(video_path, video_name), num_frames)
                kx_temp.extend(image.tolist())

            ###########
            kx.extend(kx_temp)
            ky.extend([label]*len(kx_temp))

    kx = np.array(kx)
    ky = np.array(ky)
    ky = ky[:, np.newaxis]

    if is_training:
        np.save('/home/gaolili/deepLearning_project/action_classification_adapt_DDK/data/train_npy/input_16_4_shuffle.npy', kx)
        np.save('/home/gaolili/deepLearning_project/action_classification_adapt_DDK/data/train_npy/label_16_4_shuffle.npy', ky)
    else:
        np.save('/home/gaolili/deepLearning_project/action_classification_adapt_DDK/data/test_npy/input_16_4_shuffle.npy', kx)
        np.save('/home/gaolili/deepLearning_project/action_classification_adapt_DDK/data/test_npy/label_16_4_shuffle.npy', ky)

    return kx, ky#, data_indexs


def generate_train_test_video(data_root, save_root):

    data_dict = defaultdict(lambda :defaultdict(lambda :[]))
    for root, dirs, _ in os.walk(data_root):
        for dir in dirs:
            print("dir{}".format(dir))
            video_path = os.path.join(root, dir)
            for video_name in os.listdir(video_path):
                group = video_name.split('_')[2]
                data_dict[dir][group].append(video_name)
    random.seed(100)
    for class_name, group_name in data_dict.items():
        save_train_path = os.path.join(save_root, 'train', class_name)
        if not os.path.exists(save_train_path):
            os.mkdir(save_train_path)

        save_test_path = os.path.join(save_root, 'test', class_name)
        if not os.path.exists(save_test_path):
            os.mkdir(save_test_path)

        for group_name, video_names in group_name.items():
            video_names.sort()
            random.shuffle(video_names)
            video_name_train = video_names[:int(len(video_names)*0.8)]
            video_name_test = video_names[int(len(video_names) * 0.8):]
            for video_name in video_name_train:
                shutil.copy(os.path.join(data_root, class_name, video_name), save_train_path)
            for video_name in video_name_test:
                shutil.copy(os.path.join(data_root, class_name, video_name), save_test_path)

#####################噪声添加###################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def alpha_beta_init_generate(mean, std):
    #np.random.seed(6)
    #data = np.random.normal(mean, std)
    data = 0
    while(data<=0):
        data = np.random.normal(mean, std)
    return data

def alpha_beta_add_noise(m, alpha=True, beta=True):
    if alpha:

        alpha_mean = m.alpha.data.cpu().numpy()
        alpha_noise_all = torch.empty(alpha_mean.shape[0]).to(device)
        alpha_std = 0.01873 * 5.1
        for i in range(alpha_mean.shape[0]):
            alpha_noise = alpha_beta_init_generate(alpha_mean[i], alpha_std)
            alpha_noise_all[i] = alpha_noise
        m.alpha.data = alpha_noise_all


    if beta:
        beta_mean = m.beta.data.cpu().numpy()
        beta_noise_all = torch.empty(beta_mean.shape[0]).to(device)
        beta_std = 0.05411 * 5.1
        for i in range(beta_mean.shape[0]):
            beta_noise = alpha_beta_init_generate(beta_mean[i], beta_std)
            beta_noise_all[i] = beta_noise
        m.beta.data = beta_noise_all
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

def filter_extreme_MAD(data,n, percent): #MAD:中位数去极值
      median = np.quantile(data,percent)
      new_median = np.quantile(np.abs((data - median)),percent)
      max_range = median + n*new_median
      min_range = median - n*new_median
      return min_range, max_range
def filter_extreme_3sigma(series,n=3): #3 sigma
  mean = series.mean()
  std = series.std()
  max_range = mean + n*std
  min_range = mean - n*std
  return np.clip(series,min_range,max_range)

def filter_extreme_3sigma(series,n=3): #3 sigma
  mean = series.mean()
  std = series.std()
  max_range = mean + n*std
  min_range = mean - n*std
  return min_range,max_range
def filter_extreme_percentile(series,min = 0.10,max = 0.90): #百分位法
  series = series.sort_values()
  q = series.quantile([min,max])
  return np.clip(series,q.iloc[0],q.iloc[1])

def w_add_noise(module, noise_mean, noise_std, map_min, map_max, clip_min=None, clip_max=None):
    w = module.weight.data
    if clip_min != None and clip_max != None:
        w.clamp_(clip_min, clip_max)
    w_noise = weight_map(w, noise_mean, noise_std, map_min, map_max)
    module.weight.data = w_noise




if __name__ =='__main__':
    #######################直接生成kx, ky太慢，打算先获取RGB图片################################
    data_root = '/home/gaolili/data/UCF101_sports'
    img_save_root = '/home/gaolili/data/UCF101_sports_imgs_1'
    save_root = '/home/gaolili/deepLearning_project/action_classification_adapt_DDK/data'
    label_map = {'Archery':0, 'BalanceBeam':1, 'Basketball':2, 'Bowling':3, 'GolfSwing':4, 'PushUps':5, 'Skiing':6, 'TennisSwing':7,
                 'TrampolineJumping':8, 'YoYo':9}
    #######min_frames=29, max_frames=641
    #min_frames, max_frames = compute_MinVideoFrame(data_root)
    #data_root = '/home/gaolili/data/example'
    #generate_train_test_video(data_root, save_root)
    #########每类视频随机打乱#############
    data_root_train = '/home/gaolili/deepLearning_project/action_classification_adapt_DDK_V4_new/data/UCF-10/train'
    data_root_test = '/home/gaolili/deepLearning_project/action_classification_adapt_DDK_V4_new/data/UCF-10/test'
    print("start generate testing data")
    generate_pytorch_data_shuffle(data_root_test, label_map, is_training=False)
    print("start generate training data")
    generate_pytorch_data_shuffle(data_root_train, label_map, is_training = True)
