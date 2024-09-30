import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def load_train(data_path):
    '''
    :param ：train data path
    :return: ndarray with shape (n_samples, n_features + 1 property)
    '''
    data = pd.read_csv(data_path,
                       header=None,
                       sep=',',
                       encoding='utf-8')
    print('Number of training samples: {}, Number of components: {}'.format(data.shape[0], data.shape[1] - 1))
    return np.array(data)


def load_validate(data_path):
    '''
    :param ：validate data path
    :return: ndarray with shape (n_samples, n_components + 1 property)
    '''
    data = pd.read_csv(data_path,
                       header=None,
                       sep=',',
                       encoding='utf-8')
    print('Number of testing samples: {}, Number of components: {}'.format(data.shape[0], data.shape[1]-1))
    return np.array(data)


class MyData(Dataset):
    def __init__(self, data, input_dim, train=True):
        '''
        :param data: ndarray with shape (n_samples, n_features+1 property)
        :param input_dim: number of input features (components)
        :param train: boolean indicating for loading training set or validation set
        '''
        self.input_dim = input_dim
        self.data = torch.tensor(data[:, :self.input_dim],dtype=torch.float32)
        self.GT = torch.tensor(data[:, self.input_dim],dtype=torch.float32)
        self.label = torch.tensor([ 500 <= self.GT[id] <= 600 for id in range(len(self))]).float()
        self.train = train
        self.posidx = [id for id in range(len(self)) if self.label[id] == 1.]
        self.negidx = [id for id in range(len(self)) if self.label[id] == 0.]

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.label[idx]
        if self.train:
            if label:
                pos = self.data[random.choice(self.posidx)]
                neg = self.data[random.choice(self.negidx)]
            else:
                pos = self.data[random.choice(self.negidx)]
                neg = self.data[random.choice(self.posidx)]
            feature, pos, neg = self.perbulation(1, 0.001, feature), self.perbulation(1, 0.001, pos), self.perbulation(1,
                                                                                                                     0.001,
                                                                                                                     neg)
            return self.normalize(feature), self.normalize(pos), self.normalize(neg)
        else:
            feature = self.normalize(feature)
            return feature,label

    def __len__(self):
        return len(self.data)

    def perbulation(self, miu, sigma, feature):
        '''
        :param miu: mean value of the perbulation distribution
        :param sigma: standard deviation of the percolation distribution
        :param feature: input feature of dimension 19
        :return: augmentation feature
        '''
        perb = torch.normal(miu, sigma, [self.input_dim])
        augmentation = feature * perb
        return augmentation

    def normalize(self, feature):
        '''
        :param feature: input feature of dimension 19
        :return: normalized feature with summation of all components equal to 1
        '''
        normalized_feature = feature #/ sum(feature)
        return normalized_feature




# data = load_train('./data/train_tg.csv')
# traindata = MyData(data, 19,train=True)
# testdata = MyData(data, 19,train=False)
# train_loader = DataLoader(traindata, batch_size=2)
# test_loader = DataLoader(testdata, batch_size=2)

