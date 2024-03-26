# Constants.
import torch

# Train/Test sets share the same speakers. They contain different utterances.
# 0.8 means 20% of the utterances of each speaker will be held out and placed in the test set.
TRAIN_TEST_RATIO = 0.8

CHECKPOINTS_SOFTMAX_DIR = '/home/gaolili/deepLearning_project/VoxNet.pytorch-master/checkpoints/checkpoints-softmax_3Dconv'
loss_accuracy_DIR = '/home/gaolili/deepLearning_project/VoxNet.pytorch-master/checkpoints/loss_accuracy'
###############softmax的batchsize#####################
BATCH_SIZE = 16


warm_up_epochs = 5
max_num_epochs = 500

learning_rate = 0.001

momentum = 0.5
epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#########resume:checkpoint的路径######
resume_softmax = ""
