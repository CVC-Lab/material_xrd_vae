#########################################
## Evaluate Models
#########################################
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import load_material_data,load_material_data_train_test_split
from model.SOSFlowNet import SOSFlowVAE
from model.GMVAENet import GMVAENet
from model.NormalizingFlowNet import VAENF,PlanarTransformation
from model.baselines import SimpleVAE,MLP
from model.RNNVAENet import RNNVAENet
from dataset import ndarrayDataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device("cuda:3")
model = RNNVAENet(36, 36, 20, 36, 5, 4, device)
#model = VAENF(3600, 40, PlanarTransformation, 40, 10)
#model = SimpleVAE(3600, 200, 40)
#model = MLP(3600,200,40)
#model = build_model(3600, 40, 3, 1, 3)
#model = SOSFlowVAE(3600, 200, 40, 7, 3, 1, 3, device)
model.load_state_dict(torch.load('checkpoints/RNN_VAE_3100.pth'))
model.to(device)

data_location = "/mnt/storage/tmwang/Materials/MP_v1.mat"
X,id,atom_type,y = load_material_data(data_location)
l = np.array(atom_type - 1, dtype=int)
X_train, X_test, l_train, l_test, y_train, y_test, id_train, id_test = train_test_split(X, l, y, id, test_size=0.20, random_state=9)
train_dataset = ndarrayDataset(X_train,y_train)
#train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)
test_dataset = ndarrayDataset(X_test,y_test)
#test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False)


# mapes = 0.
# max_mapes = 0.
# rowlist = []
# for batch_idx, (data, y) in enumerate(train_dataset):
#     data = data.to(device)
#     y = y.view(-1,1).to(device)
#     #y_pred = model(data)
#     _,_,_,_,y_pred = model(data)
#     mape = torch.abs((y-y_pred)/y).item()
#     rowlist.append(dict({
#         "material_id" : id_train[batch_idx],
#         "cohesive eng.":y.cpu().numpy()[0][0],
#         "predicted cohesive eng.":y_pred.detach().cpu().numpy()[0],
#         "mape":mape
#     }))
#     max_mapes = max(max_mapes, mape)
#     mapes += mape

# print("Max mape: %.4f  Average mape: %.4f" % (max_mapes, mapes/len(train_dataset)))

# df = pd.DataFrame(rowlist, columns= ["material_id", "cohesive eng.", "predicted cohesive eng.", "mape"])
# df.to_csv('results/train_VAE.csv', index=False)

# mapes = 0
# max_mapes = 0
# rowlist = []
# for batch_idx, (data, y) in enumerate(test_dataset):
#     data = data.to(device)
#     y = y.view(-1,1).to(device)
#     #y_pred = model(data)
#     _,_,_,_,y_pred = model(data)
#     mape = torch.abs((y-y_pred)/y).item()
#     rowlist.append(dict({
#         "material_id": id_test[batch_idx],
#         "cohesive eng.": y.cpu().numpy()[0][0],
#         "predicted cohesive eng.": y_pred.detach().cpu().numpy()[0],
#         "mape": mape
#     }))
#     max_mapes = max(max_mapes, mape)
#     mapes += mape

# print("Max mape: %.4f  Average mape: %.4f" % (max_mapes, mapes/len(test_dataset)))

# df = pd.DataFrame(rowlist, columns= ["material_id", "cohesive eng.", "predicted cohesive eng.", "mape"])
# df.to_csv('results/test_VAE.csv', index=False)

# model.eval()
idx = 21
mat = X_test[0,:].reshape(100,36).T
data = test_dataset[idx][0]
data = data.view(1,100,36)
print(data.shape)
data = data.to(device)
model.to(device)

#x_rec, _,_, energy = model(data, energy=True)
x_rec, _  = model(data, data, teaching_ratio=0.0)

#print("Energy: %.4f; Prediction: %.4f" % (y_test[0], energy))

recon_data = x_rec.view(-1).detach().cpu().numpy()

print(np.min(X_test[0,:]))

zmat = recon_data.reshape(36,100)

# fig, ax = plt.subplots(nrows=6, ncols=6)

# count = 0
# for row in ax:
#     for col in row:
#         col.plot(mat[count,:],color='b')
#         col.plot(zmat[count,:],color='r')
#         count += 1
angle = np.linspace(0,180,3600, endpoint=False)
plt.title("RNN_VAE")
plt.plot(angle,recon_data, label = 'predicted')
print(data.view(-1).cpu().numpy()[900:910])
plt.plot(angle,data.view(-1).cpu().numpy(), label = 'ground truth')
plt.legend()
plt.show()
plt.savefig('results/XRD_vector_RNNVAE_3100.png')

