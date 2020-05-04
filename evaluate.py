#########################################
## Evaluate Models
#########################################
import random
import numpy as numpy
import torch
from torch.utils.data import DataLoader
from utils import load_material_data_train_test_split
from SOSFlow import build_model
from model.GMVAENet import GMVAENet
from model.NormalizingFlowNet import VAENF,PlanarTransformation
from model.baselines import SimpleVAE
from dataset import ndarrayDataset
from matplotlib import pyplot as plt

device = torch.device("cuda:0")
#model = VAENF(3600, 40, PlanarTransformation, 40, 4)
#model = SimpleVAE(3600, 200, 40)
model = build_model(3600, 40, 3, 1, 3)
model.load_state_dict(torch.load('checkpoints/SOSflow.pth'))

data_location = "/mnt/storage/tmwang/Materials/MP.mat"
_,X_test,_,y_test = load_material_data_train_test_split(data_location)
test_dataset = ndarrayDataset(X_test,y_test)

model.eval()

mat = X_test[0,:].reshape(36,100)
data = test_dataset[0][0]
data = data.view(1,-1)
print(data.shape)
data = data.to(device)
model.to(device)

x_rec, _ = model(data)

recon_data = x_rec.detach().cpu().numpy()


zmat = recon_data.reshape(36,100)

fig, ax = plt.subplots(nrows=6, ncols=6)

count = 0
for row in ax:
    for col in row:
        col.plot(mat[:,count],color='b')
        col.plot(zmat[:,count],color='r')
        count += 1

plt.show()
plt.savefig('results/XRD_recon_VAE_20.png')

