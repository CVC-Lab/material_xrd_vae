import argparse
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from utils import load_material_data_train_test_split_v2
from RNN_VAE import RNNVAE
from dataset import ndarrayDataset
from globarvar import DATA_LOCATION_V2 as data_location

#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='PyTorch Implementation of RNN-VAE')

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
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=512, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=1024, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='learning rate (default: 0.001)')

# useless config right now
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=7,
                    help='number of classes (default: 7)')
parser.add_argument('--hidden_size', default=20, type=int,
                    help='gru (default: 20)')
parser.add_argument('--flow_dim', default=10, type=int,
                    help='gaussian size (default: 20)')
parser.add_argument('--input_size', default=36, type=int,
                    help='input size (default: 3600)')
parser.add_argument('--n-blocks', default=2, type=int,
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
X_train,X_test,y_train,y_test = load_material_data_train_test_split_v2(data_location)
X_train = np.reshape(X_train, (-1,100,36))
X_test = np.reshape(X_test, (-1,100,36))
train_dataset = ndarrayDataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
train_losses = np.zeros((args.epochs))
test_dataset = ndarrayDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size_val)
test_losses = np.zeros((args.epochs))

args.label_dim = y_train.shape[1]

#########################################################
## Train and Test Model
#########################################################
rnnvae = RNNVAE(args)

## Training Phase
history_loss = rnnvae.train(train_loader, test_loader)

with open('checkpoints/rnnvae_%d.npz' % args.epochs,'wb') as f:
  np.savez(f, 
  train_loss = history_loss['train_history_err'], 
  test_loss = history_loss['val_history_err'], 
  train_smape = history_loss['smape_train'],
  test_smape = history_loss['smape_test'])
torch.save(rnnvae.network.state_dict(), 'checkpoints/RNN_VAE_%d.pth' % args.epochs)

