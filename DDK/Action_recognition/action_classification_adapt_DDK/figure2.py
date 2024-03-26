import numpy as np
import os
root_path = 'D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\result'
acc_127 = []
precision_127 = []
recall_127 = []
f1_score_127 = []
eer_127 = []

acc_512 = []
precision_512 = []
recall_512 = []
f1_score_512 = []
eer_512 = []

acc_1024 = []
precision_1024 = []
recall_1024 = []
f1_score_1024 = []
eer_1024 = []

acc_2048 = []
precision_2048 = []
recall_2048 = []
f1_score_2048 = []
eer_2048 = []

acc_raw = []
precision_raw = []
recall_raw = []
f1_score_raw = []
eer_raw = []
for root, dirs, _ in os.walk(root_path):
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):

            data_max = 0
            data_min = float('inf')
            file_path = os.path.join(root, dir, file)
            for line in open(file_path).readlines():
                temp = float(line.split(' ')[-1])
                if temp > data_max:
                    data_max = temp
                if temp < data_min:
                    data_min = temp
            if 'acc' in file and '127' in file:
                acc_127.append(data_max)
            elif 'recall' in file and '127' in file:
                recall_127.append(data_max)
            elif 'precision' in file and '127' in file:
                precision_127.append(data_max)
            elif 'f1_score' in file and '127' in file:
                f1_score_127.append(data_max)

            elif 'acc' in file and '512' in file:
                acc_512.append(data_max)
            elif 'recall' in file and '512' in file:
                recall_512.append(data_max)
            elif 'precision' in file and '512' in file:
                precision_512.append(data_max)
            elif 'f1_score' in file and '512' in file:
                f1_score_512.append(data_max)

            elif 'acc' in file and '1024' in file:
                acc_1024.append(data_max)
            elif 'recall' in file and '1024' in file:
                recall_1024.append(data_max)
            elif 'precision' in file and '1024' in file:
                precision_1024.append(data_max)
            elif 'f1_score' in file and '1024' in file:
                f1_score_1024.append(data_max)

            elif 'acc' in file and '2048' in file:
                acc_2048.append(data_max)
            elif 'recall' in file and '2048' in file:
                recall_2048.append(data_max)
            elif 'precision' in file and '2048' in file:
                precision_2048.append(data_max)
            elif 'f1_score' in file and '2048' in file:
                f1_score_2048.append(data_max)

            if 'acc' in file and 'rawWaveform' in file:
                acc_raw.append(data_max)
            elif 'recall' in file and 'rawWaveform' in file:
                recall_raw.append(data_max)
            elif 'precision' in file and 'rawWaveform' in file:
                precision_raw.append(data_max)
            elif 'f1_score' in file and 'rawWaveform' in file:
                f1_score_raw.append(data_max)

            if 'accs' in file and 'raw' in file:
                acc_raw.append(data_max)
            elif 'tprs' in file and 'raw' in file:
                recall_raw.append(data_max)
            elif 'eers' in file and 'raw' in file:
                eer_raw.append(data_min)
            elif 'fms' in file and 'raw' in file:
                f1_score_raw.append(data_max)


            if 'accs' in file and '127' in file:
                acc_127.append(data_max)
            elif 'tprs' in file and '127' in file:
                recall_127.append(data_max)
            elif 'eers' in file and '127' in file:
                eer_127.append(data_min)
            elif 'fms' in file and '127' in file:
                f1_score_127.append(data_max)

            elif 'accs' in file and '512' in file:
                acc_512.append(data_max)
            elif 'tprs' in file and '512' in file:
                recall_512.append(data_max)
            elif 'eers' in file and '512' in file:
                eer_512.append(data_min)
            elif 'fms' in file and '512' in file:
                f1_score_512.append(data_max)

            elif 'accs' in file and '1024' in file:
                acc_1024.append(data_max)
            elif 'tprs' in file and '1024' in file:
                recall_1024.append(data_max)
            elif 'eers' in file and '1024' in file:
                eer_1024.append(data_min)
            elif 'fms' in file and '1024' in file:
                f1_score_1024.append(data_max)

            elif 'accs' in file and '2048' in file:
                acc_2048.append(data_max)
            elif 'tprs' in file and '2048' in file:
                recall_2048.append(data_max)
            elif 'eers' in file and '2048' in file:
                eer_2048.append(data_min)
            elif 'fms' in file and '2048' in file:
                f1_score_2048.append(data_max)


acc_127_max = max(acc_127)
acc_127_mean = np.mean(acc_127)
acc_127_std = np.std(acc_127)
if len(precision_127)!=0:
    precision_127_max = max(precision_127)
    precision_127_mean = np.mean(precision_127)
    precision_127_std = np.std(precision_127)
if len(eer_127)!=0:
    eer_127_min = min(eer_127)
    eer_127_mean = np.mean(eer_127)
    eer_127_std = np.std(eer_127)

recall_127_max = max(recall_127)
recall_127_mean = np.mean(recall_127)
recall_127_std = np.std(recall_127)

f1_score_127_max = max(f1_score_127)
f1_score_127_mean = np.mean(f1_score_127)
f1_score_127_std = np.std(f1_score_127)


acc_512_max = max(acc_512)
acc_512_mean = np.mean(acc_512)
acc_512_std = np.std(acc_512)
if len(precision_512)!=0:
    precision_512_max = max(precision_512)
    precision_512_mean = np.mean(precision_512)
    precision_512_std = np.std(precision_512)

if len(eer_512)!=0:
    eer_512_min = min(eer_512)
    eer_512_mean = np.mean(eer_512)
    eer_512_std = np.std(eer_512)

recall_512_max = max(recall_512)
recall_512_mean = np.mean(recall_512)
recall_512_std = np.std(recall_512)

f1_score_512_max = max(f1_score_512)
f1_score_512_mean = np.mean(f1_score_512)
f1_score_512_std = np.std(f1_score_512)


acc_1024_max = max(acc_1024)
acc_1024_mean = np.mean(acc_1024)
acc_1024_std = np.std(acc_1024)

if len(precision_1024)!=0:
    precision_1024_max = max(precision_1024)
    precision_1024_mean = np.mean(precision_1024)
    precision_1024_std = np.std(precision_1024)

if len(eer_1024)!=0:
    eer_1024_min = min(eer_1024)
    eer_1024_mean = np.mean(eer_1024)
    eer_1024_std = np.std(eer_1024)

recall_1024_max = max(recall_1024)
recall_1024_mean = np.mean(recall_1024)
recall_1024_std = np.std(recall_1024)

f1_score_1024_max = max(f1_score_1024)
f1_score_1024_mean = np.mean(f1_score_1024)
f1_score_1024_std = np.std(f1_score_1024)

if len(acc_2048)!=0:
    acc_2048_max = max(acc_2048)
    acc_2048_mean = np.mean(acc_2048)
    acc_2048_std = np.std(acc_2048)
if len(precision_2048)!=0:
    precision_2048_max = max(precision_2048)
    precision_2048_mean = np.mean(precision_2048)
    precision_2048_std = np.std(precision_2048)

if len(eer_2048)!=0:
    eer_2048_min = min(eer_2048)
    eer_2048_mean = np.mean(eer_2048)
    eer_2048_std = np.std(eer_2048)
if len(recall_2048)!=0:
    recall_2048_max = max(recall_2048)
    recall_2048_mean = np.mean(recall_2048)
    recall_2048_std = np.std(recall_2048)
if len(f1_score_2048)!=0:
    f1_score_2048_max = max(f1_score_2048)
    f1_score_2048_mean = np.mean(f1_score_2048)
    f1_score_2048_std = np.std(f1_score_2048)

if len(precision_raw)!=0:
    acc_raw_max = max(acc_raw)
    precision_raw_max = max(precision_raw)
    recall_raw_max = max(recall_raw)
    f1_score_raw_max = max(f1_score_raw)

    acc_raw_mean = np.mean(acc_raw)
    precision_raw_mean = np.mean(precision_raw)
    recall_raw_mean = np.mean(recall_raw)
    f1_score_raw_mean = np.mean(f1_score_raw)

    acc_raw_std = np.std(acc_raw)
    precision_raw_std = np.std(precision_raw)
    recall_raw_std = np.std(recall_raw)
    f1_score_raw_std = np.std(f1_score_raw)
if len(eer_raw)!=0:
    acc_raw_max = max(acc_raw)
    eer_raw_min = min(eer_raw)
    recall_raw_max = max(recall_raw)
    f1_score_raw_max = max(f1_score_raw)

    acc_raw_mean = np.mean(acc_raw)
    eer_raw_mean = np.mean(eer_raw)
    recall_raw_mean = np.mean(recall_raw)
    f1_score_raw_mean = np.mean(f1_score_raw)

    acc_raw_std = np.std(acc_raw)
    eer_raw_std = np.std(eer_raw)
    recall_raw_std = np.std(recall_raw)
    f1_score_raw_std = np.std(f1_score_raw)
a = 0