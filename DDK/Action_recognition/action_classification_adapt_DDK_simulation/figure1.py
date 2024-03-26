import matplotlib.pyplot as plt
import numpy as np
############nfft=127######################
###############pre-training######################
'''
test_acc_127_fft_conv = []
test_acc_127_fft_mfcc = []
test_acc_127_fft_hf = []

test_loss_127_fft_conv = []
test_loss_127_fft_mfcc = []
test_loss_127_fft_hf = []

file_test_acc_127_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft1270.919.txt')
file_test_acc_127_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc127_0.898.txt')
file_test_acc_127_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf127_0.917.txt')

file_test_loss_127_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft1270.919.txt')
file_test_loss_127_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc127_0.898.txt')
file_test_loss_127_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf127_0.917.txt')

for line in file_test_acc_127_fft_conv.readlines():
    test_acc_127_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_acc_127_fft_mfcc.readlines():
    test_acc_127_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_acc_127_fft_hf.readlines():
    test_acc_127_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_127_fft_conv.readlines():
    test_loss_127_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_127_fft_mfcc.readlines():
    test_loss_127_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_127_fft_hf.readlines():
    test_loss_127_fft_hf.append(float(line.split('\n')[0]))

iterations_len = min(len(test_acc_127_fft_conv), len(test_acc_127_fft_mfcc), len(test_acc_127_fft_hf))
iterations = [i for i in range(0,iterations_len*10, 10)]
plt.figure()
plt.plot(iterations, test_acc_127_fft_conv[0:iterations_len], label="test_acc_127_fft_conv")
plt.plot(iterations, test_acc_127_fft_mfcc[0:iterations_len], label="test_acc_127_fft_mfcc")
plt.plot(iterations, test_acc_127_fft_hf[0:iterations_len], label="test_acc_127_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_127\\pre-training\\test_acc.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(iterations, test_loss_127_fft_conv[0:iterations_len], label="test_loss_127_fft_conv")
plt.plot(iterations, test_loss_127_fft_mfcc[0:iterations_len], label="test_loss_127_fft_mfcc")
plt.plot(iterations, test_loss_127_fft_hf[0:iterations_len], label="test_loss_127_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_127\\pre-training\\test_loss.png', bbox_inches='tight')
plt.show()
'''
####################triplet-training############################
'''
test_eer_127_fft_conv = []
test_eer_127_fft_mfcc = []
test_eer_127_fft_hf = []

test_loss_127_fft_conv = []
test_loss_127_fft_mfcc = []
test_loss_127_fft_hf = []
file_test_eer_127_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_127.txt')
file_test_loss_127_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_127.txt')

file_test_eer_127_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc127.txt')
file_test_loss_127_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc127.txt')

file_test_eer_127_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf127.txt')
file_test_loss_127_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf127.txt')

for line in file_test_eer_127_fft_conv.readlines():
    test_eer_127_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_eer_127_fft_mfcc.readlines():
    test_eer_127_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_eer_127_fft_hf.readlines():
    test_eer_127_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_127_fft_conv.readlines():
    test_loss_127_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_127_fft_mfcc.readlines():
    test_loss_127_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_127_fft_hf.readlines():
    test_loss_127_fft_hf.append(float(line.split('\n')[0]))

plt.figure(figsize=(20,20))
ax = plt.subplot(311)
iterations_127 = [i for i in range(0,len(test_loss_127_fft_conv)*10, 10)]
ax.plot(iterations_127, test_eer_127_fft_conv, label="test_eer_127")
ax.plot(iterations_127, test_loss_127_fft_conv, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_conv/loss_fft_conv')

ax = plt.subplot(312)
iterations_127 = [i for i in range(0,len(test_loss_127_fft_mfcc)*10, 10)]
ax.plot(iterations_127, test_eer_127_fft_mfcc, label="test_eer_127")
ax.plot(iterations_127, test_loss_127_fft_mfcc, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_mfcc/loss_fft_mfcc')


ax = plt.subplot(313)
iterations_127 = [i for i in range(0,len(test_loss_127_fft_hf)*10, 10)]
ax.plot(iterations_127, test_eer_127_fft_hf, label="test_eer_127")
ax.plot(iterations_127, test_loss_127_fft_hf, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_hf/loss_fft_hf')



plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_127\\triplet-training\\test_all.png', bbox_inches='tight')
plt.show()
'''

############nfft=512######################
###############pre-training######################
'''
test_acc_512_fft_conv = []
test_acc_512_fft_mfcc = []
test_acc_512_fft_hf = []

test_loss_512_fft_conv = []
test_loss_512_fft_mfcc = []
test_loss_512_fft_hf = []

file_test_acc_512_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft5120.924.txt')
file_test_acc_512_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc512_0.944.txt')
file_test_acc_512_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf512_0.938.txt')

file_test_loss_512_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft5120.924.txt')
file_test_loss_512_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc512_0.944.txt')
file_test_loss_512_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf512_0.938.txt')

for line in file_test_acc_512_fft_conv.readlines():
    test_acc_512_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_acc_512_fft_mfcc.readlines():
    test_acc_512_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_acc_512_fft_hf.readlines():
    test_acc_512_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_512_fft_conv.readlines():
    test_loss_512_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_512_fft_mfcc.readlines():
    test_loss_512_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_512_fft_hf.readlines():
    test_loss_512_fft_hf.append(float(line.split('\n')[0]))

iterations_len = min(len(test_acc_512_fft_conv), len(test_acc_512_fft_mfcc), len(test_acc_512_fft_hf))
iterations = [i for i in range(0,iterations_len*10, 10)]
plt.figure()
plt.plot(iterations, test_acc_512_fft_conv[0:iterations_len], label="test_acc_512_fft_conv")
plt.plot(iterations, test_acc_512_fft_mfcc[0:iterations_len], label="test_acc_512_fft_mfcc")
plt.plot(iterations, test_acc_512_fft_hf[0:iterations_len], label="test_acc_512_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_512\\pre-training\\test_acc.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(iterations, test_loss_512_fft_conv[0:iterations_len], label="test_loss_512_fft_conv")
plt.plot(iterations, test_loss_512_fft_mfcc[0:iterations_len], label="test_loss_512_fft_mfcc")
plt.plot(iterations, test_loss_512_fft_hf[0:iterations_len], label="test_loss_512_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_512\\pre-training\\test_loss.png', bbox_inches='tight')
plt.show()
'''
####################triplet-training############################
'''
test_eer_512_fft_conv = []
test_eer_512_fft_mfcc = []
test_eer_512_fft_hf = []

test_loss_512_fft_conv = []
test_loss_512_fft_mfcc = []
test_loss_512_fft_hf = []
file_test_eer_512_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_512.txt')
file_test_loss_512_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_512.txt')

file_test_eer_512_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc512.txt')
file_test_loss_512_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc512.txt')

file_test_eer_512_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf512.txt')
file_test_loss_512_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf512.txt')

for line in file_test_eer_512_fft_conv.readlines():
    test_eer_512_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_eer_512_fft_mfcc.readlines():
    test_eer_512_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_eer_512_fft_hf.readlines():
    test_eer_512_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_512_fft_conv.readlines():
    test_loss_512_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_512_fft_mfcc.readlines():
    test_loss_512_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_512_fft_hf.readlines():
    test_loss_512_fft_hf.append(float(line.split('\n')[0]))

plt.figure(figsize=(20,20))
ax = plt.subplot(311)
iterations_512 = [i for i in range(0,len(test_loss_512_fft_conv)*10, 10)]
ax.plot(iterations_512, test_eer_512_fft_conv, label="test_eer_512")
ax.plot(iterations_512, test_loss_512_fft_conv, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_conv/loss_fft_conv')

ax = plt.subplot(312)
iterations_512 = [i for i in range(0,len(test_loss_512_fft_mfcc)*10, 10)]
ax.plot(iterations_512, test_eer_512_fft_mfcc, label="test_eer_512")
ax.plot(iterations_512, test_loss_512_fft_mfcc, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_mfcc/loss_fft_mfcc')


ax = plt.subplot(313)
iterations_512 = [i for i in range(0,len(test_loss_512_fft_hf)*10, 10)]
ax.plot(iterations_512, test_eer_512_fft_hf, label="test_eer_512")
ax.plot(iterations_512, test_loss_512_fft_hf, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_hf/loss_fft_hf')



plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_512\\triplet-training\\test_all.png', bbox_inches='tight')
plt.show()
'''

############nfft=1024######################
###############pre-training######################
'''
test_acc_1024_fft_conv = []
test_acc_1024_fft_mfcc = []
test_acc_1024_fft_hf = []

test_loss_1024_fft_conv = []
test_loss_1024_fft_mfcc = []
test_loss_1024_fft_hf = []

file_test_acc_1024_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft10240.939.txt')
file_test_acc_1024_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc_nfft1024_0.916.txt')
file_test_acc_1024_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf_nfft1024_a0.01b1_0.934.txt')

file_test_loss_1024_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft10240.939.txt')
file_test_loss_1024_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc_nfft1024_0.916.txt')
file_test_loss_1024_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf_nfft1024_a0.01b1_0.934.txt')
for line in file_test_acc_1024_fft_conv.readlines():
    test_acc_1024_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_acc_1024_fft_mfcc.readlines():
    test_acc_1024_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_acc_1024_fft_hf.readlines():
    test_acc_1024_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_1024_fft_conv.readlines():
    test_loss_1024_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_1024_fft_mfcc.readlines():
    test_loss_1024_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_1024_fft_hf.readlines():
    test_loss_1024_fft_hf.append(float(line.split('\n')[0]))

iterations_len = min(len(test_acc_1024_fft_conv), len(test_acc_1024_fft_mfcc), len(test_acc_1024_fft_hf))
iterations = [i for i in range(0,iterations_len*10, 10)]
plt.figure()
plt.plot(iterations, test_acc_1024_fft_conv[0:iterations_len], label="test_acc_1024_fft_conv")
plt.plot(iterations, test_acc_1024_fft_mfcc[0:iterations_len], label="test_acc_1024_fft_mfcc")
plt.plot(iterations, test_acc_1024_fft_hf[0:iterations_len], label="test_acc_1024_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_1024\\pre-training\\test_acc.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(iterations, test_loss_1024_fft_conv[0:iterations_len], label="test_loss_1024_fft_conv")
plt.plot(iterations, test_loss_1024_fft_mfcc[0:iterations_len], label="test_loss_1024_fft_mfcc")
plt.plot(iterations, test_loss_1024_fft_hf[0:iterations_len], label="test_loss_1024_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_1024\\pre-training\\test_loss.png', bbox_inches='tight')
plt.show()
'''
####################triplet-training############################
'''
test_eer_1024_fft_conv = []
test_eer_1024_fft_mfcc = []
test_eer_1024_fft_hf = []

test_loss_1024_fft_conv = []
test_loss_1024_fft_mfcc = []
test_loss_1024_fft_hf = []
file_test_eer_1024_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_1024.txt')
file_test_loss_1024_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_1024.txt')

file_test_eer_1024_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc_nfft1024.txt')
file_test_loss_1024_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc_nfft1024.txt')

file_test_eer_1024_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf_nfft1024_a0.01b1.txt')
file_test_loss_1024_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf_nfft1024_a0.01b1.txt')

for line in file_test_eer_1024_fft_conv.readlines():
    test_eer_1024_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_eer_1024_fft_mfcc.readlines():
    test_eer_1024_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_eer_1024_fft_hf.readlines():
    test_eer_1024_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_1024_fft_conv.readlines():
    test_loss_1024_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_1024_fft_mfcc.readlines():
    test_loss_1024_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_1024_fft_hf.readlines():
    test_loss_1024_fft_hf.append(float(line.split('\n')[0]))

plt.figure(figsize=(20,20))
ax = plt.subplot(311)
iterations_1024 = [i for i in range(0,len(test_loss_1024_fft_conv)*10, 10)]
ax.plot(iterations_1024, test_eer_1024_fft_conv, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024_fft_conv, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_conv/loss_fft_conv')

ax = plt.subplot(312)
iterations_1024 = [i for i in range(0,len(test_loss_1024_fft_mfcc)*10, 10)]
ax.plot(iterations_1024, test_eer_1024_fft_mfcc, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024_fft_mfcc, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_mfcc/loss_fft_mfcc')


ax = plt.subplot(313)
iterations_1024 = [i for i in range(0,len(test_loss_1024_fft_hf)*10, 10)]
ax.plot(iterations_1024, test_eer_1024_fft_hf, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024_fft_hf, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_hf/loss_fft_hf')



plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_1024\\triplet-training\\test_all.png', bbox_inches='tight')
plt.show()
'''
############nfft=2048######################
###############pre-training######################
'''
test_acc_2048_fft_conv = []
test_acc_2048_fft_mfcc = []
test_acc_2048_fft_hf = []

test_loss_2048_fft_conv = []
test_loss_2048_fft_mfcc = []
test_loss_2048_fft_hf = []

file_test_acc_2048_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft20480.925.txt')
file_test_acc_2048_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc_nfft2048_0.92.txt')
file_test_acc_2048_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf_nfft2048_a0.01b0.5_0.938.txt')

file_test_loss_2048_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft20480.925.txt')
file_test_loss_2048_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc_nfft2048_0.92.txt')
file_test_loss_2048_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf_nfft2048_a0.01b0.5_0.938.txt')
for line in file_test_acc_2048_fft_conv.readlines():
    test_acc_2048_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_acc_2048_fft_mfcc.readlines():
    test_acc_2048_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_acc_2048_fft_hf.readlines():
    test_acc_2048_fft_hf.append(float(line.split('\n')[0]))

for line in file_test_loss_2048_fft_conv.readlines():
    test_loss_2048_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_2048_fft_mfcc.readlines():
    test_loss_2048_fft_mfcc.append(float(line.split('\n')[0]))
for line in file_test_loss_2048_fft_hf.readlines():
    test_loss_2048_fft_hf.append(float(line.split('\n')[0]))

iterations_len = min(len(test_acc_2048_fft_conv), len(test_acc_2048_fft_mfcc), len(test_acc_2048_fft_hf))
iterations = [i for i in range(0,iterations_len*10, 10)]
plt.figure()
plt.plot(iterations, test_acc_2048_fft_conv[0:iterations_len], label="test_acc_2048_fft_conv")
plt.plot(iterations, test_acc_2048_fft_mfcc[0:iterations_len], label="test_acc_2048_fft_mfcc")
plt.plot(iterations, test_acc_2048_fft_hf[0:iterations_len], label="test_acc_2048_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_2048\\pre-training\\test_acc.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(iterations, test_loss_2048_fft_conv[0:iterations_len], label="test_loss_2048_fft_conv")
plt.plot(iterations, test_loss_2048_fft_mfcc[0:iterations_len], label="test_loss_2048_fft_mfcc")
plt.plot(iterations, test_loss_2048_fft_hf[0:iterations_len], label="test_loss_2048_fft_hf")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_2048\\pre-training\\test_loss.png', bbox_inches='tight')
plt.show()
'''
####################triplet-training############################
'''
test_eer_2048_fft_conv = []
test_eer_2048_fft_mfcc = []
#test_eer_2048_fft_hf = []

test_loss_2048_fft_conv = []
test_loss_2048_fft_mfcc = []
#test_loss_2048_fft_hf = []
#file_test_eer_2048_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_2048.txt')
#file_test_loss_2048_fft_conv = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_2048.txt')

file_test_eer_2048_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc_nfft2048.txt')
file_test_loss_2048_fft_mfcc = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc_nfft2048.txt')

#file_test_eer_2048_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf_nfft1024_a0.01b1.txt')
#file_test_loss_2048_fft_hf = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf_nfft1024_a0.01b1.txt')

#for line in file_test_eer_2048_fft_conv.readlines():
#    test_eer_2048_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_eer_2048_fft_mfcc.readlines():
    test_eer_2048_fft_mfcc.append(float(line.split('\n')[0]))
#for line in file_test_eer_2048_fft_hf.readlines():
#    test_eer_2048_fft_hf.append(float(line.split('\n')[0]))

#for line in file_test_loss_2048_fft_conv.readlines():
#    test_loss_2048_fft_conv.append(float(line.split('\n')[0]))
for line in file_test_loss_2048_fft_mfcc.readlines():
    test_loss_2048_fft_mfcc.append(float(line.split('\n')[0]))
#for line in file_test_loss_2048_fft_hf.readlines():
#    test_loss_2048_fft_hf.append(float(line.split('\n')[0]))

plt.figure(figsize=(20,20))
ax = plt.subplot(211)
iterations_2048 = [i for i in range(0,len(test_loss_2048_fft_conv)*10, 10)]
ax.plot(iterations_2048, test_eer_2048_fft_conv, label="test_eer_2048")
ax.plot(iterations_2048, test_loss_2048_fft_conv, c='r', label="test_loss_2048")
ax.set_title('nff=2048')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_conv/loss_fft_conv')

ax = plt.subplot(212)
iterations_2048 = [i for i in range(0,len(test_loss_2048_fft_mfcc)*10, 10)]
ax.plot(iterations_2048, test_eer_2048_fft_mfcc, label="test_eer_2048")
ax.plot(iterations_2048, test_loss_2048_fft_mfcc, c='r', label="test_loss_2048")
ax.set_title('nff=2048')
ax.set_xlabel('epoch')
ax.set_ylabel('eer_fft_mfcc/loss_fft_mfcc')


#ax = plt.subplot(313)
#iterations_1024 = [i for i in range(0,len(test_loss_1024_fft_hf)*10, 10)]
#ax.plot(iterations_1024, test_eer_1024_fft_hf, label="test_eer_1024")
#ax.plot(iterations_1024, test_loss_1024_fft_hf, c='r', label="test_loss_1024")
#ax.set_title('nff=1024')
#ax.set_xlabel('epoch')
#ax.set_ylabel('eer_fft_hf/loss_fft_hf')



plt.savefig('D:\\deeplearning_project\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\figure\\nfft_2048\\triplet-training\\test_all.png', bbox_inches='tight')
plt.show()
'''


#########################speaker 10类小样本训练######################
import scipy.io
############hf#################
epoch_hf = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\hf\\result\\epoch_hf_0.98.mat')
loss_hf = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\hf\\result\\epoch_loss_hf_0.98.mat')
eopch_hf = epoch_hf['epoch'].tolist()[0]
loss_hf = loss_hf['loss'].tolist()[0]
##########fft+hf################
epoch_fft_hf = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\hf\\result\\epoch_hf_nfft8_0.99.mat')
loss_fft_hf = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\hf\\result\\epoch_loss_nfft8_0.99.mat')
eopch_fft_hf = epoch_fft_hf['epoch'].tolist()[0]
loss_fft_hf = loss_fft_hf['loss'].tolist()[0]
##########fft+mfcc###################
epoch_mfcc = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\mfcc\\result\\epoch_mfcc40_epoch.mat')
loss_mfcc = scipy.io.loadmat('D:\\Voiceprint_20201027\\Input_file\\speaker\\10s\\mat\\mfcc\\result\\epoch_mfcc40_loss.mat')
eopch_mfcc = epoch_mfcc['epoch'].tolist()[0]
loss_mfcc = loss_mfcc['loss'].tolist()[0]
########fft+conv###################
test_loss = []
iterations = []
file_test_loss = open('D:\\Voiceprint_20201027\\convNet_pytorch\\result\\losses_speaker_0.884.txt')
file_iterations = open('D:\\Voiceprint_20201027\\convNet_pytorch\\result\\iteration_speaker_0.884.txt')

for line in file_test_loss.readlines():
    test_loss.append(float(line.split('\n')[0]))
for line in file_iterations.readlines():
    iterations.append(float(line.split('\n')[0]))
plt.figure(figsize=(20,20))

ax = plt.subplot(411)
ax.plot(eopch_hf, loss_hf, label="hf")
ax.set_title('hf')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax = plt.subplot(412)
ax.plot(eopch_fft_hf, loss_fft_hf, label="fft+hf")
ax.set_title('fft+hf')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax = plt.subplot(413)
ax.plot(eopch_mfcc, loss_mfcc, label="mfcc")
ax.set_title('fft+mfcc')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax = plt.subplot(414)
ax.plot(iterations, test_loss, label="fft+conv")
ax.set_title('fft+conv')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')


plt.savefig('D:\\Voiceprint_20201027\\figure\\speaker\\test_all.png', bbox_inches='tight')
plt.show()

#########################whale 4类小样本训练######################
import scipy.io
########fft+conv###################
test_loss_whale = []
iterations_whale = []
file_iterations = open('D:\\Voiceprint_20201027\\convNet_pytorch\\result\\iteration_whale_0.928.txt')
file_test_loss = open('D:\\Voiceprint_20201027\\convNet_pytorch\\result\\losses_whale_0.928.txt')

for line in file_test_loss.readlines():
    test_loss_whale.append(float(line.split('\n')[0]))
for line in file_iterations.readlines():
    iterations_whale.append(float(line.split('\n')[0]))
plt.figure()
plt.plot(iterations_whale, test_loss_whale, label="test_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('D:\\Voiceprint_20201027\\figure\\whale\\test_loss.png', bbox_inches='tight')
plt.show()