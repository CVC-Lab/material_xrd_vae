# TODO: Merge model evaluation in one single file

import argparse
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from utils import load_material_data_train_test_split
from SOSFlow import SOSFlow
from dataset import ndarrayDataset

#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='SOSFlow')

## Dataset
parser.add_argument('--dataset', type=str, choices=['mnist'],
                    default='mnist', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=100,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch-size-val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning-rate', default=1e-3, type=float,
                    help='learning rate (default: 0.001)')

# useless config right now
parser.add_argument('--decay-epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr-decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num-classes', type=int, default=7,
                    help='number of classes (default: 7)')
parser.add_argument('--hidden-size', default=40, type=int,
                    help='hidden dim size (default: 40)')
parser.add_argument('--input-size', default=3600, type=int,
                    help='input size (default: 3600)')

parser.add_argument('--num-polynomials', type=int, default=2,
                    help='number of polynomials (default: 5)')
parser.add_argument('--degree', default=1, type=int,
                    help='polynomial degrees (default: 3)')
parser.add_argument('--n-blocks', default=2, type=int,
                    help='number of blocks (default: 4)')

## Others
parser.add_argument('--verbose', default=0, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')

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
sosflow = SOSFlow(args)

## Training Phase
history_loss = sosflow.train(train_loader, test_loader)

# with open('checkpoints/GMVAE.npz','wb') as f:
#   np.savez(f, train_acc = history_loss['train_history_acc'], test_acc = history_loss['val_history_acc'])