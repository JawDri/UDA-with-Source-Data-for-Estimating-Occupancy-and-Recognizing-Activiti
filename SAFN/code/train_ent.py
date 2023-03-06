import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,Dataset
from utils import OfficeImage, weights_init, print_args
from model import ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--source", default="")
parser.add_argument("--target", default="")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--class_num", default=2)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--weight_entropy", default=0.1)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
args = parser.parse_args()
print_args(args)

import pandas as pd
import numpy as np
Source_train = pd.read_csv("/content/drive/MyDrive/SAFN/code/data/Source_train.csv")
Source_test = pd.read_csv("/content/drive/MyDrive/SAFN/code/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/SAFN/code/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/SAFN/code/data/Target_test.csv")

FEATURES = list(i for i in Source_train.columns if i!= 'labels')
TARGET = "labels"

from sklearn.preprocessing import StandardScaler
Normarizescaler = StandardScaler()
Normarizescaler.fit(np.array(Source_train[FEATURES]))

class PytorchDataSet(Dataset):
    
    def __init__(self, df):
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return {"X":self.train_X[idx], "Y":self.train_Y[idx]}

Source_train = PytorchDataSet(Source_train)
Source_test = PytorchDataSet(Source_test)
Target_train = PytorchDataSet(Target_train)
Target_test = PytorchDataSet(Target_test)

source_loader = torch.utils.data.DataLoader(Source_train, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(Target_train, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResBase50().cuda()
netF = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cuda()
netF.apply(weights_init)


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return args.weight_L2norm * l

def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return args.weight_entropy * (entropy / float(p_softmax.size(0)))   
 

opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))


    for i, t_img in tqdm.tqdm(enumerate(target_loader_iter)):
        t_imgs = t_img['X']
        try:
            s_img = next(source_loader_iter)
        except:
            source_loader_iter = iter(source_loader)
            s_img = next(source_loader_iter)
        s_imgs = s_img['X']
        s_labels = s_img['Y']

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())     
        t_imgs = Variable(t_imgs.cuda())
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        s_bottleneck = netG(s_imgs)
        t_bottleneck = netG(t_imgs)
        s_fc2_emb, s_logit = netF(s_bottleneck)
        t_fc2_emb, t_logit = netF(t_bottleneck)

        s_cls_loss = get_cls_loss(s_logit, s_labels)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)
        
        t_prob = F.softmax(t_logit)
        t_entropy_loss = get_entropy_loss(t_prob)

        loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss + t_entropy_loss
        loss.backward()
        
        if (i+1) % 5 == 0:
            
            print("cls_loss: %.4f, entropy_loss: %.4f" % (s_cls_loss.data.item(), t_entropy_loss.data.item()))
            #print("cls_loss: {0}, entropy_loss: {1}".format(s_cls_loss.data[0], t_entropy_loss.data[0]))
        opt_g.step()
        opt_f.step()
    
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), os.path.join(args.snapshot, "Office31_IAFN_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))
        torch.save(netF.state_dict(), os.path.join(args.snapshot, "Office31_IAFN_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth"))

