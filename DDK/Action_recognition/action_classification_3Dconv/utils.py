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

def read_video(video_path):

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
    id = 0
    num_frames = 16
    resize_divisor = 2
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
                    while(success == False):
                        j += 1
                        video.set(cv2.CAP_PROP_POS_FRAMES, j)
                        success, frame = video.read()

                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    image = cv2.resize(frame_gray, (int(width / resize_divisor), int(height / resize_divisor)), interpolation=cv2.INTER_CUBIC)
                    image_flatten = np.reshape(image, (int(width / resize_divisor) * int(height / resize_divisor)))
                    images[is_read, :] = image_flatten
                    is_read += 1
                    id = j

            id += 1
    return images

def read_video_RGB(video_path):

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
    id = 0
    num_frames = 16

    images = np.zeros((num_frames, 112, 112, 3))

    if video_frames <= num_frames:

        while id < video_frames:
            success, frame = video.read()
            if success == True:
                image = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)

                ###############随机crop#######################
                '''
                image = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_CUBIC)
                width_offset = random.randint(0, 128 - 112 - 1)
                height_offset = random.randint(0, 171 - 112 - 1)
                image = image[:,width_offset:width_offset+112, height_offset:height_offset+112, :]
                '''
                images[id, :, :, :] = image
            id += 1
    else:
        is_read = 0
        pad = int(video_frames / num_frames)
        while id < video_frames:

            if not id % pad and is_read < num_frames:
                video.set(cv2.CAP_PROP_POS_FRAMES, id)
                success, frame = video.read()
                #print(id)
                if success == True:
                    image = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
                    ###############随机crop#######################
                    '''
                    image = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_CUBIC)
                    width_offset = random.randint(0, 128 - 112 - 1)
                    height_offset = random.randint(0, 171 - 112 - 1)
                    image = image[width_offset:width_offset + 112, height_offset:height_offset + 112, :]
                    '''
                    images[is_read, :, :, :] = image

                    is_read += 1

                else:
                    j = id + 1
                    video.set(cv2.CAP_PROP_POS_FRAMES, j)
                    success, frame = video.read()
                    while(success == False):
                        j += 1
                        video.set(cv2.CAP_PROP_POS_FRAMES, j)
                        success, frame = video.read()
                    image = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_CUBIC)
                    ###############随机crop#######################
                    '''
                    image = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_CUBIC)
                    width_offset = random.randint(0, 128 - 112 - 1)
                    height_offset = random.randint(0, 171 - 112 - 1)
                    image = image[:, width_offset:width_offset + 112, height_offset:height_offset + 112, :]
                    '''
                    images[is_read, :, :, :] = image
                    is_read += 1
                    id = j

            id += 1
    return images

def generate_pytorch_data(data_root, label_map, is_training=True):
    kx = []
    ky = []
    for root, dirs, _ in os.walk(data_root):
        for dir in dirs:
            print("dir{}".format(dir))
            label = label_map[dir]
            video_path = os.path.join(root, dir)
            for video_name in os.listdir(video_path):
                print("video_name{}".format(video_name))
                image = read_video_RGB(os.path.join(video_path, video_name))
                kx.append(image.tolist())
                ky.append(label)
    kx = np.array(kx)
    ky = np.array(ky)
    ky = ky[:, np.newaxis]

    if is_training:
        np.save('/home/gaolili/deepLearning_project/action_classification_3Dconv/data/train_npy/input_16_112_112_resize.npy', kx)
        np.save('/home/gaolili/deepLearning_project/action_classification_3Dconv/data/train_npy/label_16_112_112_resize.npy', ky)
    else:
        np.save('/home/gaolili/deepLearning_project/action_classification_3Dconv/data/test_npy/input_16_112_112_resize.npy', kx)
        np.save('/home/gaolili/deepLearning_project/action_classification_3Dconv/data/test_npy/label_16_112_112_resize.npy', ky)
    return kx, ky

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


if __name__ =='__main__':
    #######################直接生成kx, ky太慢，打算先获取RGB图片，目前还没调试################################
    data_root = '/home/gaolili/data/UCF101_sports'
    img_save_root = '/home/gaolili/data/UCF101_sports_imgs_1'
    save_root = '/home/gaolili/deepLearning_project/action_classification_3Dconv/data'
    label_map = {'Archery':0, 'BalanceBeam':1, 'Basketball':2, 'Bowling':3, 'GolfSwing':4, 'PushUps':5, 'Skiing':6, 'TennisSwing':7,
                 'TrampolineJumping':8, 'YoYo':9}
    #######min_frames=29, max_frames=641
    #min_frames, max_frames = compute_MinVideoFrame(data_root)
    #data_root = '/home/gaolili/data/example'
    #generate_train_test_video(data_root, save_root)
    data_root_train = '/home/gaolili/deepLearning_project/action_classification_3Dconv/data/train'
    data_root_test = '/home/gaolili/deepLearning_project/action_classification_3Dconv/data/test'
    generate_pytorch_data(data_root_test, label_map, is_training=False)
    generate_pytorch_data(data_root_train, label_map, is_training = True)
