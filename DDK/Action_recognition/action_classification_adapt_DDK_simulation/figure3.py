import numpy as np
import os
root_path = 'D:\\Voiceprint_20201027\\fft512_convNet_pytorch_speaker\\result'
acc= []
precision = []
recall = []
f1_score = []
eer = []
alpha = []
beta = []
for root, dirs, _ in os.walk(root_path):
    for dir in dirs:
        for file in os.listdir(os.path.join(root, dir)):

            data_max = 0
            data_min = float('inf')
            file_path = os.path.join(root, dir, file)
            if 'alpha' in file:
                alpha.append(float(open(file_path).readlines()[-1]))

            elif 'beta' in file:
                beta.append(float(open(file_path).readlines()[-1]))

            else:
                for line in open(file_path).readlines():
                    temp = float(line.split(' ')[-1])
                    if temp > data_max:
                        data_max = temp
                    if temp < data_min:
                        data_min = temp
                if 'accuracy' in file:
                    acc.append(data_max)
                elif 'recall' in file:
                    recall.append(data_max)
                elif 'precision' in file:
                    precision.append(data_max)
                elif 'f1_score' in file:
                    f1_score.append(data_max)


acc_max = max(acc)
acc_mean = np.mean(acc)
acc_std = np.std(acc)

precision_max = max(precision)
precision_mean = np.mean(precision)
precision_std = np.std(precision)


recall_max = max(recall)
recall_mean = np.mean(recall)
recall_std = np.std(recall)

f1_score_max = max(f1_score)
f1_score_mean = np.mean(f1_score)
f1_score_std = np.std(f1_score)



a = 0