import os
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, optimizer, train_dataloader, valid_dataloader, args, seed_id, noise_mean = None, noise_std = None, map_min = None, map_max = None):
    best_f1 = 0
    logger.info('Start Training!')
    for epoch in range(1, args.epochs+1):
        model.train()
        print("model.DDK.alpha", model.DDK.alpha)
        print("model.DDK.beta", model.DDK.beta)
        ############一个epoch, 参数添加噪声， 这里不需要再添加噪声了####################
        #for m in model.modules():
        #    if isinstance(m, nn.Linear):
        #        w_add_noise(m, noise_mean, noise_std, map_min, map_max)


        for step, (x, y) in enumerate(train_dataloader):
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, args)
            loss = F.cross_entropy(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if (step+1) % 200 == 0:
                logger.info(f'|EPOCHS| {epoch:>}/{args.epochs} |STEP| {step+1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}')

        avg_loss, accuracy, _, _, f1, _ = evaluate(model, valid_dataloader, args, noise_mean=0.20329, noise_std=1.14726, map_min=40, map_max=300)
        logger.info('-'*50)
        logger.info(f'|* VALID SET *| |VAL LOSS| {avg_loss:>.4f} |ACC| {accuracy:>.4f} |F1| {f1:>.4f}')
        logger.info('-'*50)

        if f1 > best_f1:
            best_f1 = f1
            logger.info(f'Saving best model... F1 score is {best_f1:>.4f}')
            if not os.path.isdir(args.model_save_path):
                os.mkdir(args.model_save_path)
            torch.save(model.state_dict(), os.path.join(args.model_save_path, str(seed_id)+"_best.pt"))
            logger.info('Model saved!')


def evaluate(model, valid_dataloader, args, noise_mean = None, noise_std = None, map_min = None, map_max = None):
    ############一个epoch, 参数添加噪声####################
    #######添加噪声测试，使用的是添加噪声后的权重进行下次训练，相当于给下一次训练添加了噪声
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w_add_noise(m, noise_mean, noise_std, map_min, map_max)

    ############################################
    with torch.no_grad():
        model.eval()
        losses, correct = 0, 0
        y_hats, targets = [], []

        for x, y in valid_dataloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x, args)
            loss = F.cross_entropy(pred, y)
            losses += loss.item()

            y_hat = torch.max(pred, 1)[1]
            y_hats += y_hat.tolist()
            targets += y.tolist()
            correct += (y_hat == y).sum().item()

    avg_loss, accuracy, precision, recall, f1, cm = metrics(valid_dataloader, losses, correct, y_hats, targets)
    return avg_loss, accuracy, precision, recall, f1, cm