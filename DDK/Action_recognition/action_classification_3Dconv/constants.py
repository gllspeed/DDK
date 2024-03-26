# Constants.
import torch

CHECKPOINTS_SOFTMAX_DIR = '/home/gaolili/deepLearning_project/action_classification_3Dconv/checkpoints/dataset_112_112/checkpoints-softmax'
loss_accuracy_DIR = '/home/gaolili/deepLearning_project/action_classification_3Dconv/checkpoints/dataset_112_112/loss_accuracy'
###############softmax的batchsize#####################
#BATCH_SIZE = 43 * 3  # have to be a multiple of 3.效果最好
BATCH_SIZE = 16#6944#9920

warm_up_epochs = 5
max_num_epochs = 200
###########learning_rate
learning_rate = 0.003

momentum = 0.5
epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#########resume:checkpoint的路径######
resume_softmax = ""

