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

def screen(net, memory_loader, test_loader,epoch,args):
    net.eval()
    total_top1,  total_num, feature_bank = 0.0,  0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_loader:
            feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous() #[38301,dim]
        feature_labels = memory_loader.dataset.label.to(device, non_blocking=True)
        posidx = memory_loader.dataset.valid_id
        pos_features = feature_bank[posidx]
        pos_center = pos_features.mean(dim=0,keepdim=True)
        negidx = memory_loader.dataset.invalid_id
        neg_features = feature_bank[negidx]
        neg_center = neg_features.mean(dim=0,keepdim=True)
        
        test_bank = []
        test_bar = tqdm(test_loader)
        for data in test_bar:
            test_bar.set_description('Screen Epoch: [{}/{}] '.format(epoch, args.epochs))
            data = data.to(device, non_blocking=True)
            feature = net(data) #[bs,dim]
            test_bank.append(feature)
        test_bank = torch.cat(test_bank, dim=0).contiguous()  # [num_val,dim]
        pos_score = torch.mm(pos_center, test_bank.T).squeeze() #[num_val]
        neg_score = torch.mm(neg_center, test_bank.T).squeeze()  # [num_val]
        score = (pos_score - neg_score)/ (pos_score + torch.abs(neg_score) )
        sim_weight, sim_indices = score.topk(k=10, dim=-1)  # [bs,top-k]
    return np.array(sim_indices.cpu())

