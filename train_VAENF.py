import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from utils import load_material_data_train_test_split
from NormalizingFlow import NormalizingFlow
from dataset import ndarrayDataset

#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='PyTorch Implementation of DGM Clustering')

## Used only in notebooks
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

## Dataset
parser.add_argument('--dataset', type=str, choices=['mnist'],
                    default='mnist', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=20,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-5, type=float,
                    help='learning rate (default: 0.001)')

# useless config right now
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=7,
                    help='number of classes (default: 7)')
parser.add_argument('--hidden_size', default=40, type=int,
                    help='gaussian size (default: 20)')
parser.add_argument('--input_size', default=3600, type=int,
                    help='input size (default: 3600)')
parser.add_argument('--n-blocks', default=10, type=int,
                    help='number of blocks (default: 4)')

args = parser.parse_args()
## Random Seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
  torch.cuda.manual_seed(SEED)

########################################################
## Data Loader
########################################################
data_location = "/mnt/storage/tmwang/Materials/MP.mat"
X_train,X_test,y_train,y_test = load_material_data_train_test_split(data_location)
train_dataset = ndarrayDataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
train_losses = np.zeros((args.epochs))
test_dataset = ndarrayDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size_val)
test_losses = np.zeros((args.epochs))

args.input_size = np.prod(train_dataset[0][0].size())
print(args.input_size)
#########################################################
## Train and Test Model
#########################################################
vaenf = NormalizingFlow(args)

## Training Phase
history_loss = vaenf.train(train_loader, test_loader)

with open('checkpoints/vaenf.npz','wb') as f:
  np.savez(f, train_loss = history_loss['train_history_err'], test_loss = history_loss['val_history_err'])
torch.save(vaenf.network.state_dict(), 'checkpoints/VAENF_%d.pth' % args.epochs)

