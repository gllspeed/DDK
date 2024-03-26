# Constants.
import torch


CHECKPOINTS_SOFTMAX_DIR = '/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK_simulation/checkpoints/DDK_noise_training/checkpoints-softmax_DDK'
loss_accuracy_DIR = '/home/gaolili/deepLearning_project/3Dvoxel_classification_adapt_DDK_simulation/checkpoints/DDK_noise_training/loss_accuracy'
###############softmax的batchsize#####################

BATCH_SIZE_TRAIN = 64#25600
BATCH_SIZE_TEST = 6400

num_frames = 32
warm_up_epochs = 5
max_num_epochs = 500

#########学习率######
learning_rate = 0.01

momentum = 0.5
epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#########resume:checkpoint的路径######
resume_softmax = ""

