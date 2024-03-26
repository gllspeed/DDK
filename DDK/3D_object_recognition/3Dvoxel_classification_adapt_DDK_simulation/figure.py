import matplotlib.pyplot as plt
import numpy as np
###########fft+Conv#################

'''
################pre-training##############
iterations = [i for i in range(0,370, 10)]
test_acc_512 = []
test_acc_127 = []
test_acc_1024 = []
test_acc_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_acc_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft5120.924.txt')
file_test_acc_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft1270.919.txt')
file_test_acc_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft10240.939.txt')
file_test_acc_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_acc_fftConv_nfft20480.925.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft5120.924.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft1270.919.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft10240.939.txt')
file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\test\\test_loss_fftConv_nfft20480.925.txt')

for line in file_test_acc_512.readlines():
    test_acc_512.append(float(line.split('\n')[0]))

for line in file_test_acc_127.readlines():
    test_acc_127.append(float(line.split('\n')[0]))

for line in file_test_acc_1024.readlines():
    test_acc_1024.append(float(line.split('\n')[0]))

for line in file_test_acc_2048.readlines():
    test_acc_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

for line in file_test_loss_2048.readlines():
    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_acc_512[0:37], label="test_acc_512")
plt.plot(iterations, test_acc_127[0:37], label="test_acc_127")
plt.plot(iterations, test_acc_1024[0:37], label="test_acc_1024")
plt.plot(iterations, test_acc_2048[0:37], label="test_acc_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\figure\\test_acc.png', bbox_inches='tight')
plt.show()


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_loss_512[0:37], label="test_loss_512")
plt.plot(iterations, test_loss_127[0:37], label="test_loss_127")
plt.plot(iterations, test_loss_1024[0:37], label="test_loss_1024")
plt.plot(iterations, test_loss_2048[0:37], label="test_loss_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\pre-training\\figure\\test_loss.png', bbox_inches='tight')
plt.show()
'''
'''
################triplet-training##############
test_eer_512 = []
test_eer_127 = []
test_eer_1024 = []
test_eer_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_eer_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_512.txt')
file_test_eer_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_127.txt')
file_test_eer_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_1024.txt')
#file_test_eer_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_eers_fftConv_2048.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_512.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_127.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_1024.txt')
file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\test\\test_loss_fftConv_2048.txt')

for line in file_test_eer_512.readlines():
    test_eer_512.append(float(line.split('\n')[0]))

for line in file_test_eer_127.readlines():
    test_eer_127.append(float(line.split('\n')[0]))

for line in file_test_eer_1024.readlines():
    test_eer_1024.append(float(line.split('\n')[0]))

#for line in file_test_eer_2048.readlines():
#    test_eer_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

for line in file_test_loss_2048.readlines():
    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure(figsize=(20,20))
ax = plt.subplot(411)
iterations_127 = [i for i in range(0,len(test_loss_127)*10, 10)]
ax.plot(iterations_127, test_eer_127, label="test_eer_127")
ax.plot(iterations_127, test_loss_127, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

ax = plt.subplot(412)
iterations_512 = [i for i in range(0,len(test_loss_512)*10, 10)]
ax.plot(iterations_512, test_eer_512, label="test_eer_512")
ax.plot(iterations_512, test_loss_512, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')


ax = plt.subplot(413)
iterations_1024 = [i for i in range(0,len(test_loss_1024)*10, 10)]
ax.plot(iterations_1024, test_eer_1024, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

ax = plt.subplot(414)
iterations_2048 = [i for i in range(0,len(test_loss_2048)*10, 10)]
ax.plot(iterations_2048, test_loss_2048, c='r',label="test_loss_2048")
ax.set_title('nff=2048')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftConv\\triplet-training\\figure\\test_all.png', bbox_inches='tight')
plt.show()
'''
'''
#########################mfcc########################################

##############pre-training##############################
iterations = [i for i in range(0,400, 10)]
test_acc_512 = []
test_acc_127 = []
test_acc_1024 = []
test_acc_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_acc_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc512_0.944.txt')
file_test_acc_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc127_0.898.txt')
file_test_acc_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc_nfft1024_0.916.txt')
file_test_acc_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_acc_mfcc_nfft2048_0.92.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc512_0.944.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc127_0.898.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc_nfft1024_0.916.txt')
file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\test\\test_loss_mfcc_nfft2048_0.92.txt')

for line in file_test_acc_512.readlines():
    test_acc_512.append(float(line.split('\n')[0]))

for line in file_test_acc_127.readlines():
    test_acc_127.append(float(line.split('\n')[0]))

for line in file_test_acc_1024.readlines():
    test_acc_1024.append(float(line.split('\n')[0]))

for line in file_test_acc_2048.readlines():
    test_acc_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

for line in file_test_loss_2048.readlines():
    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_acc_512[0:40], label="test_acc_512")
plt.plot(iterations, test_acc_127[0:40], label="test_acc_127")
plt.plot(iterations, test_acc_1024[0:40], label="test_acc_1024")
plt.plot(iterations, test_acc_2048[0:40], label="test_acc_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\figure\\test_acc.png', bbox_inches='tight')
plt.show()


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_loss_512[0:40], label="test_loss_512")
plt.plot(iterations, test_loss_127[0:40], label="test_loss_127")
plt.plot(iterations, test_loss_1024[0:40], label="test_loss_1024")
plt.plot(iterations, test_loss_2048[0:40], label="test_loss_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\mfcc\\figure\\test_loss.png', bbox_inches='tight')
plt.show()
'''
'''
################triplet-training##############
test_eer_512 = []
test_eer_127 = []
test_eer_1024 = []
test_eer_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_eer_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc512.txt')
file_test_eer_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc127.txt')
file_test_eer_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc_nfft1024.txt')
file_test_eer_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_eers_mfcc_nfft2048.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc512.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc127.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc_nfft1024.txt')
file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\test\\test_loss_mfcc_nfft2048.txt')

for line in file_test_eer_512.readlines():
    test_eer_512.append(float(line.split('\n')[0]))

for line in file_test_eer_127.readlines():
    test_eer_127.append(float(line.split('\n')[0]))

for line in file_test_eer_1024.readlines():
    test_eer_1024.append(float(line.split('\n')[0]))

for line in file_test_eer_2048.readlines():
    test_eer_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

for line in file_test_loss_2048.readlines():
    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure(figsize=(20,20))
ax = plt.subplot(411)
iterations_127 = [i for i in range(0,len(test_loss_127)*10, 10)]
ax.plot(iterations_127, test_eer_127, label="test_eer_127")
ax.plot(iterations_127, test_loss_127, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

ax = plt.subplot(412)
iterations_512 = [i for i in range(0,len(test_loss_512)*10, 10)]
ax.plot(iterations_512, test_eer_512, label="test_eer_512")
ax.plot(iterations_512, test_loss_512, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')


ax = plt.subplot(413)
iterations_1024 = [i for i in range(0,len(test_loss_1024)*10, 10)]
ax.plot(iterations_1024, test_eer_1024, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

ax = plt.subplot(414)
iterations_2048 = [i for i in range(0,len(test_loss_2048)*10, 10)]
ax.plot(iterations_2048, test_eer_2048, label="test_eer_2048")
ax.plot(iterations_2048, test_loss_2048, c='r',label="test_loss_2048")
ax.set_title('nff=2048')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\mfcc\\figure\\test_all.png', bbox_inches='tight')
plt.show()
'''

#########################HF################################

'''
#################HF pre-training#########################

iterations = [i for i in range(0,310, 10)]
test_acc_512 = []
test_acc_127 = []
test_acc_1024 = []
test_acc_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_acc_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf512_0.938.txt')
file_test_acc_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf127_0.917.txt')
file_test_acc_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf_nfft1024_a0.01b1_0.934.txt')
file_test_acc_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_acc_hf_nfft2048_a0.01b0.5_0.938.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf512_0.938.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf127_0.917.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf_nfft1024_a0.01b1_0.934.txt')
file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\test\\test_loss_hf_nfft2048_a0.01b0.5_0.938.txt')

for line in file_test_acc_512.readlines():
    test_acc_512.append(float(line.split('\n')[0]))

for line in file_test_acc_127.readlines():
    test_acc_127.append(float(line.split('\n')[0]))

for line in file_test_acc_1024.readlines():
    test_acc_1024.append(float(line.split('\n')[0]))

for line in file_test_acc_2048.readlines():
    test_acc_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

for line in file_test_loss_2048.readlines():
    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_acc_512[0:31], label="test_acc_512")
plt.plot(iterations, test_acc_127[0:31], label="test_acc_127")
plt.plot(iterations, test_acc_1024[0:31], label="test_acc_1024")
plt.plot(iterations, test_acc_2048[0:31], label="test_acc_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\figure\\test_acc.png', bbox_inches='tight')
plt.show()


plt.figure()
#plt.plot(iterations, accuracy, 'bv-', label="accuracy")
#plt.plot(iterations, accuracy_2048, label="accuracy_nfft_2048")
#plt.plot(iterations, accuracy_1024, label="accuracy_nfft_1024")
#plt.plot(iterations, accuracy_512, label="accuracy_nfft_512")
#plt.plot(iterations[0:101], accuracy_256[0:101], label="accuracy_nfft_256")
#plt.plot(iterations[0:101], accuracy_128[0:101], label="accuracy_nfft_128")

plt.plot(iterations, test_loss_512[0:31], label="test_loss_512")
plt.plot(iterations, test_loss_127[0:31], label="test_loss_127")
plt.plot(iterations, test_loss_1024[0:31], label="test_loss_1024")
plt.plot(iterations, test_loss_2048[0:31], label="test_loss_2048")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\pre-training\\hf\\figure\\test_loss.png', bbox_inches='tight')
plt.show()
'''

################triplet-training##############
test_eer_512 = []
test_eer_127 = []
test_eer_1024 = []
test_eer_2048 = []
test_loss_127 = []
test_loss_512 = []
test_loss_1024 = []
test_loss_2048 = []

file_test_eer_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf512.txt')
file_test_eer_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf127.txt')
file_test_eer_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_hf_nfft1024_a0.01b1.txt')
#file_test_eer_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_eers_mfcc_nfft2048.txt')

file_test_loss_512 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf512.txt')
file_test_loss_127 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf127.txt')
file_test_loss_1024 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_hf_nfft1024_a0.01b1.txt')
#file_test_loss_2048 = open('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\test\\test_loss_mfcc_nfft2048.txt')

for line in file_test_eer_512.readlines():
    test_eer_512.append(float(line.split('\n')[0]))

for line in file_test_eer_127.readlines():
    test_eer_127.append(float(line.split('\n')[0]))

for line in file_test_eer_1024.readlines():
    test_eer_1024.append(float(line.split('\n')[0]))

#for line in file_test_eer_2048.readlines():
#    test_eer_2048.append(float(line.split('\n')[0]))


for line in file_test_loss_512.readlines():
    test_loss_512.append(float(line.split('\n')[0]))

for line in file_test_loss_127.readlines():
    test_loss_127.append(float(line.split('\n')[0]))

for line in file_test_loss_1024.readlines():
    test_loss_1024.append(float(line.split('\n')[0]))

#for line in file_test_loss_2048.readlines():
#    test_loss_2048.append(float(line.split('\n')[0]))


plt.figure(figsize=(20,20))
ax = plt.subplot(311)
iterations_127 = [i for i in range(0,len(test_loss_127)*10, 10)]
ax.plot(iterations_127, test_eer_127, label="test_eer_127")
ax.plot(iterations_127, test_loss_127, c='r', label="test_loss_127")
ax.set_title('nff=127')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

ax = plt.subplot(312)
iterations_512 = [i for i in range(0,len(test_loss_512)*10, 10)]
ax.plot(iterations_512, test_eer_512, label="test_eer_512")
ax.plot(iterations_512, test_loss_512, c='r', label="test_loss_512")
ax.set_title('nff=512')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')


ax = plt.subplot(313)
iterations_1024 = [i for i in range(0,len(test_loss_1024)*10, 10)]
ax.plot(iterations_1024, test_eer_1024, label="test_eer_1024")
ax.plot(iterations_1024, test_loss_1024, c='r', label="test_loss_1024")
ax.set_title('nff=1024')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')

'''
ax = plt.subplot(414)
iterations_2048 = [i for i in range(0,len(test_loss_2048)*10, 10)]
ax.plot(iterations_2048, test_eer_2048, label="test_eer_2048")
ax.plot(iterations_2048, test_loss_2048, c='r',label="test_loss_2048")
ax.set_title('nff=2048')
ax.set_xlabel('epoch')
ax.set_ylabel('eer/loss')
'''
plt.savefig('D:\\deeplearning_project\\deep-speaker_sitw_smallCNN_pytorch_fftHF_fftMFCC\\triplet-training\\hf\\figure\\test_all.png', bbox_inches='tight')
plt.show()