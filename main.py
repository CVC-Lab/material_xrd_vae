"""

Read and prepare data

"""

import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import ndarrayDataset
from models import MLP

data_location = "/mnt/storage/tmwang/Materials/MP.mat"

data = sio.loadmat(data_location)

input_mat = data['MP']

# count data in different classes
id = input_mat[:,0]
atom_type = input_mat[:,1]
y = input_mat[:,2] # target value
X = input_mat[:,3:] # training data
print(X.shape)
for i in range(7):
    cnt = np.count_nonzero(atom_type == (i+1))
    print("Type %d : %d" % (i+1, cnt))
print(np.max(y),np.min(y))

# First train everything
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

"""

Body part of train/test

"""

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--batch-size', type=int, default=128, metexitavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=9, metavar='S',
                    help='random seed (default: 9)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='gpu card id (default: 0)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:%d" % args.gpu if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = ndarrayDataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
train_losses = np.zeros((args.epochs))
test_dataset = ndarrayDataset(X_test,y_test)
test_loader = DataLoader(test_dataset, batch_size=1)
test_losses = np.zeros((args.epochs))

model = MLP(3600,100,1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='sum')



def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device).view(-1,1)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y, y_pred)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, y) in enumerate(test_loader):
            data = data.to(device)
            y = y.to(device).view(-1,1)
            y_pred = model(data)
            test_loss += (criterion(y, y_pred).item())

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.hist(y[np.logical_and(atom_type==2, y<10)], 100, facecolor='blue', alpha=0.5)
    plt.show()
    plt.savefig('energy_histogram.png')
    # for epoch in range(1, args.epochs + 1):
    #     train(epoch)
    #     test(epoch)
