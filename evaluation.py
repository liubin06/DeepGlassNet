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
from sklearn.metrics import roc_auc_score
import evaluation
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def eval(net, memory_loader, valid_loader):
    net.eval()
    total_top1,  total_num, feature_bank = 0.0,  0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_loader:
            feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous() #[num_train,dim]
        posidx = memory_loader.dataset.valid_id
        negidx = memory_loader.dataset.invalid_id
        pos_features = feature_bank[posidx]
        neg_features = feature_bank[negidx]
        pos_center = pos_features.mean(dim=0, keepdim=True) #[1,dim]
        neg_center = neg_features.mean(dim=0, keepdim=True)  # [1,dim]


        # loop validation data to predict the label by knn search
        valid_bar = valid_loader
        val_bank = []
        val_label = valid_loader.dataset.label.to(device, non_blocking = True)
        for data, target in valid_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data) #[bs,dim]
            val_bank.append(feature)
        val_bank = torch.cat(val_bank, dim=0).contiguous()  # [num_val,dim]
        pos_score = torch.mm(pos_center, val_bank.T).squeeze() #[num_val]
        neg_score = torch.mm(neg_center, val_bank.T).squeeze()  # [num_val]
        score = (pos_score - neg_score)/ (pos_score + neg_score )
        sim_weight, sim_indices = score.topk(k=100, dim=-1)  # [bs,top-k]
        TN = torch.sum(val_label[sim_indices]).item()
        pre = TN/100
        auc = roc_auc_score(val_label.cpu().numpy(), score.cpu().numpy())
    return pre,auc
