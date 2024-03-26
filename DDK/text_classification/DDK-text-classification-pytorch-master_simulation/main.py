import os
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from build_vocab import build_dictionary
from dataset import CustomTextDataset, collate_fn
from model import RCNN
from trainer import train, evaluate
from utils import *
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed_id):
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    #if args.n_gpu > 0:
    #    torch.cuda.manual_seed_all(args.seed)


def main(args, seed_id):
    model = RCNN(vocab_size=args.vocab_size,
                 embedding_dim=args.embedding_dim,
                 hidden_size=args.hidden_size,
                 hidden_size_linear=args.hidden_size_linear,
                 class_num=args.class_num,
                 dropout=args.dropout,
                 DDK_type = args.DDK_type).to(args.device)
    #model.load_state_dict(torch.load(
    #    '/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/model_saved_word_dim/30_best.pt'))
    model.load_state_dict(torch.load(
        '/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/model_saved_word_feature_dim/20_best.pt'))
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model, dim=0)
    ############alpha, beta的值添加噪声###########
    alpha_beta_add_noise(model.DDK)


    train_texts, train_labels = read_file(args.train_file_path)
    word2idx = build_dictionary(train_texts, vocab_size=args.vocab_size)
    logger.info('Dictionary Finished!')


    full_dataset = CustomTextDataset(train_texts, train_labels, word2idx)
    num_train_data = len(full_dataset) - args.num_val_data
    train_dataset, val_dataset = random_split(full_dataset, [num_train_data, args.num_val_data])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  collate_fn=lambda x: collate_fn(x, args),
                                  batch_size=args.batch_size,
                                  shuffle=True)

    valid_dataloader = DataLoader(dataset=val_dataset,
                                  collate_fn=lambda x: collate_fn(x, args),
                                  batch_size=args.batch_size,
                                  shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_dataloader, valid_dataloader, args, seed_id, noise_mean=0.20329, noise_std=1.14726, map_min=40,
                        map_max=300)
    logger.info('******************** Train Finished ********************')

    # Test
    if args.test_set:
        test_texts, test_labels = read_file(args.test_file_path)
        test_dataset = CustomTextDataset(test_texts, test_labels, word2idx)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=lambda x: collate_fn(x, args),
                                     batch_size=args.batch_size,
                                     shuffle=True)

        model.load_state_dict(torch.load(os.path.join(args.model_save_path, str(seed_id) +"_best.pt")))
        _, accuracy, precision, recall, f1, cm = evaluate(model, test_dataloader, args, noise_mean=0.20329, noise_std=1.14726, map_min=40,
                        map_max=300)
        logger.info('-'*50)
        logger.info(f'|* TEST SET *| |ACC| {accuracy:>.4f} |PRECISION| {precision:>.4f} |RECALL| {recall:>.4f} |F1| {f1:>.4f}')
        logger.info('-'*50)
        logger.info('---------------- CONFUSION MATRIX ----------------')
        for i in range(len(cm)):
            logger.info(cm[i])
        logger.info('--------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed', nargs='+', type=int)
    parser.add_argument('--test_set', action='store_true', default=True)

    # data
    parser.add_argument("--train_file_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/data/train.csv")
    parser.add_argument("--test_file_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master/data/test.csv")
    parser.add_argument("--model_save_path", type=str, default="/home/gaolili/deepLearning_project/DDK-text-classification-pytorch-master_simulation/model_saved_word_feature_dim")
    parser.add_argument("--num_val_data", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)

    # model
    parser.add_argument("--DDK_type", type=str, default='word_feature_dim')
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--hidden_size_linear", type=int, default=512)
    parser.add_argument("--class_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    #parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    for seed_id in args.seed:
        set_seed(seed_id)

        main(args, seed_id)