import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset,Dataset
from utils import OfficeImage, print_args
from model import ResBase50, ResClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--target", default="")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=False)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--snapshot", default="")
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--result", default="")
parser.add_argument("--class_num", default=2)
parser.add_argument("--extract", default=False)
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

t_loader = torch.utils.data.DataLoader(Target_test, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)


netG = ResBase50().cuda()
netF = ResClassifier(class_num=args.class_num, extract=False, dropout_p=args.dropout_p).cuda()
netG.eval()
netF.eval()

for epoch in range(int(args.epoch/2), args.epoch +1):
    if epoch % 10 != 0:
        continue
    netG.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_IAFN_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth")))
    netF.load_state_dict(torch.load(os.path.join(args.snapshot, "Office31_IAFN_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth")))
    correct = 0
    tick = 0
    preds_val, gts_val = [], []
    for img in t_loader:
        imgs = img['X']
        labels = img['Y']
        tick += 1
        imgs = Variable(imgs.cuda())
        pred = netF(netG(imgs))
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        correct += np.equal(labels, pred).sum()
        preds_val.extend(pred)
        gts_val.extend(labels)

    preds_val = np.asarray(preds_val)
    gts_val = np.asarray(gts_val)

    correct = correct * 1.0 / len(Target_test)
    print("Epoch {0}: {1}".format(epoch, correct))
    f1score= f1_score(gts_val, preds_val, average = 'weighted').astype(float)
    fscore= score(gts_val, preds_val)[2].astype(float)
    print('\n({}) F1-Score. v {:.4f}'.format(epoch, f1score))
    print('\n({}) F1-Score-labels. v '.format(epoch), ['{:.4f}'.format(x) for x in fscore])

    
    
    

