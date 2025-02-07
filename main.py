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
import screening
import pandas as pd
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('Torch version: {}, Gpu is available: {}'.format(torch.__version__,USE_CUDA))
torch.autograd.set_detect_anomaly(True)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def criterion(anchor,positive,negative):
    '''
    :param batch:
    :return: loss
    '''
    pos_sim = torch.sum(anchor * positive,dim=-1) #[bs,]
    neg_sim = torch.sum(anchor * negative,dim=-1) #[bs,]
    loss = -torch.log(torch.sigmoid((pos_sim-neg_sim)/args.tau)).mean()
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

def list_type(arg):
    return [int(x) for x in arg[1:-1].split(',')]




    return total_top1 / total_num * 100
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--num_workers', default=8, type=int, help='Batch size in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-7, type=float, help='Weight_decay')
    parser.add_argument('--num_components', default=18, type=int, help='Number of components')
    parser.add_argument('--interval', default=[500, 600], type=list_type, help='The interval of glass transition temperatures to be SCREENED')
    parser.add_argument('--embedding_dim', default=128, type=int, help='The dimension of the embeddings associated with each component')
    parser.add_argument('--out_dim', default=512, type=int, help='The dimension of the out put feature')
    parser.add_argument('--fm_dim', default=4, type=int, help='The dimension for factorization of the adjacency matrix')
    parser.add_argument('--num_heads', default=1, type=int, help='Number of attention heads')
    parser.add_argument('--tau', default=0.1, type=float, help='Temperature scalling for loss function')
    parser.add_argument('--noise_std', default=0.1, type=float, help='Stanrd deviriation of noise perbulation for data augmentation')
   

    init_seed(2024)
    args = parser.parse_args()
    print(args)

    ####################### Step1: Data Preparation #######################
    print('The interval of glass transition temperatures to be SCREENED:', args.interval)
    train_path = args.root + '/train_tg.csv'
    valid_path = args.root + '/validation_tg.csv'
    test_path = args.root + '/test_tg.csv'
    traindata = utils.load_train(train_path)  #load data as ndarray
    validdata = utils.load_validate(valid_path)
    testdata = utils.load_test(test_path)
    mean, std = traindata.mean(axis=0) [:args.num_components], traindata.std(axis=0)[:args.num_components]
    train_data = utils.MyData(traindata, mean, std, args.num_components, args.interval, args.noise_std, phase = 'Training')
    memor_data = utils.MyData(traindata, mean, std, args.num_components, args.interval, args.noise_std, phase = 'Evaluation') 
    valid_data = utils.MyData(validdata, mean, std, args.num_components, args.interval, args.noise_std, phase = 'Evaluation')
    test_data  = utils.MyData(testdata , mean, std, args.num_components, args.interval, args.noise_std, phase = 'Screening')
    print("Number of training samples within desired GT :{}; Number of training samples out of desired GT interval:{}".format(sum(train_data.label),len(train_data)-sum(train_data.label)))
    print("Number of validating samples within desired GT :{}; Number of validating samples out of desired GT interval:{}".format(sum(valid_data.label),len(valid_data)-sum(valid_data.label)))
    print('Number of testing samples to be SCREENED :{}'.format (len(test_data)))
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
    test_loader = DataLoader(test_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers
                              )
    component = np.array(pd.read_csv(args.root + '/result.csv',header=None,sep=',',encoding='utf-8'))


    ######################## Step2: Model Setup #######################
    model = model.DeepGlassNet(args.num_components, args.embedding_dim, args.fm_dim, args.num_heads, args.out_dim).to(device)

    ######################## Step3: Optimizer Config #######################
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ######################## Step4: Model Training #######################
    if not os.path.exists('./results'):
        os.makedirs('./results')
    result = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        if epoch % 1 == 0:
            pre,auc =evaluation.eval(model,memor_loader,valid_loader)
            result.append([pre,auc])
            print('Validation Epoch: [{}/{}]: Precision:{:.1f}%, AUC:{:.4f}' .format(epoch, args.epochs,pre*100,auc))
        if epoch % 5 == 0:
            screened_id = screening.screen(model,memor_loader,test_loader,epoch,args)
            print('Top-10 Screened Samples at Epoch: [{}/{}]'.format(epoch, args.epochs))
            predict = testdata[screened_id]
            print(predict)
    best_rest= np.array(result).max(axis=0)
    best_idx = np.array(result).argmax(axis=0)
    print('Best Result: (Precision:{} at Epoch: {}), (AUC:{:.4f} at Epoch {})'.format(best_rest[0],best_idx[0]+1,best_rest[1],best_idx[1]+1))
    print('\t')
