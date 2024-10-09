#-*- coding: utf-8 -*-
import argparse
import os
import datetime
import random
import torch
import numpy as np
from torch import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import model
import evaluation
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Torch version: {}, Gpu is available: {}'.format(torch.__version__,USE_CUDA))

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def criterion(anchor,positive,negative):
    '''
    :param batch: anchor,positive, negative features [bs,dim]
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    loss = -torch.log(torch.sigmoid(pos_sim-neg_sim)).mean()
    return loss

def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for anchor, pos_feature, neg_feature in train_bar:
        anchor, pos_feature, neg_feature = anchor.to(device, non_blocking=True), pos_feature.to(device, non_blocking=True), neg_feature.to(device, non_blocking=True)
        anchor_emb = net(anchor)
        pos_emb = net(pos_feature)
        neg_emb = net(neg_feature)

        #calculate loss value
        loss = criterion(anchor_emb,pos_emb,neg_emb)

        #optimize
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

def precision(net, memory_loader, valid_loader):
    net.eval()
    total_top1,  total_num, feature_bank = 0.0,  0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_loader:
            feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous() #[38301,dim]
        feature_labels = memory_loader.dataset.label.to(device, non_blocking=True)
        posidx = memory_loader.dataset.posidx
        pos_features = feature_bank[posidx]
        pos_center = pos_features.mean(dim=0,keepdim=True)
        negidx = memory_loader.dataset.negidx
        neg_features = feature_bank[negidx]
        neg_center = neg_features.mean(dim=0,keepdim=True)


        # loop validation data to predict the label by knn search
        valid_bar = tqdm(valid_loader)
        for data, target in valid_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data) #[bs,dim]

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [bs, 38301]
            pos_sim = torch.mm(pos_center,feature.T).squeeze()
            neg_sim = torch.mm(neg_center,feature.T).squeeze()
            pred_labels = (pos_sim > neg_sim).float()
            equal_elements = torch.eq(pred_labels, target)
            total_top1 += torch.sum(equal_elements).item()

            valid_bar.set_description('Validation Epoch: [{}/{}] Precision:{:.2f}% '
                                     .format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--num_workers', default=0, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight_decay')
    init_seed(2024)
    args = parser.parse_args()
    print(args)

    ####################### Step1: Data Preparation #######################
    train_path = args.root + '/train_tg.csv'
    valid_path = args.root + '/validation_tg.csv'
    traindata = utils.load_train(train_path)  #load data as ndarray
    validdata = utils.load_validate(valid_path)

    train_data = utils.MyData(traindata, 19, train=True) # data augmentation, data normalization, and organize data as (anchor, postive, negative) triples for training.
    memor_data = utils.MyData(traindata,19, train=False) # only data normalization, and organize training data as (data, lable) tuples for KNN validation.
    valid_data = utils.MyData(validdata, 19, train=False)# only data normalization, and organize validating data as (data, lable) tuples for KNN validation.

    train_loader = DataLoader(train_data,                        #load data as minibatch for GPU computation.
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=args.num_workers)
    memor_loader = DataLoader(memor_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers
                              )

    ######################## Step2: Model Setup #######################
    model = model.DeepGlass().to(device)

    ######################## Step3: Optimizer Config #######################
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ######################## Step4: Model Training #######################
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        if epoch % 1 == 0:
            acc = precision(model,memor_loader,valid_loader)
