import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader, TensorDataset,Dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_nn = nn.Sequential(
            nn.Linear(9, 512),#len(FEATURES)
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
            )
    
    def forward(self, x):
        logits = self.linear_nn(x)
        return logits


model=Net()

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        self.model_resnet = model

        
       
        model_resnet = self.model_resnet

        self.linear_nn = model_resnet.linear_nn
        

    def forward(self, x):
        
       
        x = self.linear_nn(x)

        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return 256


class ResClassifier(nn.Module):

    def __init__(self, class_num=5, extract=True, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 32),
            nn.BatchNorm1d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc2 = nn.Linear(32, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))            
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit         
        return logit
