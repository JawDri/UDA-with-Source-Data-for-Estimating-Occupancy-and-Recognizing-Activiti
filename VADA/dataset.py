import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.io import loadmat

import pandas as pd
Source_train = pd.read_csv("./data/Source_train.csv")
Source_test = pd.read_csv("./data/Source_test.csv")
Target_train = pd.read_csv("./data/Target_train.csv")
Target_test = pd.read_csv("./data/Target_test.csv")


class Dataset(data.Dataset):
    def __init__(self, iseval, dataratio=1.0):

        self.eval = iseval
        labels = Source_train.labels
        data = Source_train.drop(['labels'], axis= 1)
        #print(data.iloc[5])
        self.datalist_src = [{
                                'image': data.iloc[ij],
                                'label': int(labels[ij])
        } for ij in range(len(labels)) if np.random.rand() <= dataratio]

        labels = Target_train.labels
        data = Target_train.drop(['labels'], axis= 1)
        
        self.datalist_target = [{
                                'image': data.iloc[ij],
                                'label': int(labels[ij])
        } for ij in range(len(labels)) if np.random.rand() <= dataratio]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 1), (0.5, 1))

        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)

    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):

        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source = self.datalist_src[index_src]['image']
        image_source = image_source.to_numpy().reshape((912,1))#32/9
        image_source = self.totensor(image_source)
        #image_source = self.normalize(image_source)

        image_target = self.datalist_target[index_target]['image']
        image_target = image_target.to_numpy().reshape((912,1))#32/9
        image_target = self.totensor(image_target)
        #image_target = self.normalize(image_target)

        return image_source, self.datalist_src[index_src]['label'], image_target, self.datalist_target[index_target]['label']


class Dataset_eval(data.Dataset):
    def __init__(self):

        # svhn.
        labels = Target_test.labels
        data = Target_test.drop(['labels'], axis= 1)
        self.datalist_target = [{
                                'image': data.iloc[ij],
                                'label': int(labels[ij])
        } for ij in range(len(labels))]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.datalist_target)

    def __getitem__(self, index):

        image_target = self.datalist_target[index]['image']
        image_target = image_target.to_numpy().reshape((912,1))#32/9
        image_target = self.totensor(image_target)
        #image_target = self.normalize(image_target)

        return image_target, self.datalist_target[index]['label']


def GenerateIterator(args, iseval=False):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size if not iseval else args.batch_size_eval,
        'shuffle': True,
        'num_workers': args.workers,
        'drop_last': True,
    }

    return data.DataLoader(Dataset(iseval), **params)


def GenerateIterator_eval(args):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size_eval,
        'num_workers': args.workers,
    }

    return data.DataLoader(Dataset_eval(), **params)
