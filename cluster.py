from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np 
from matplotlib import pyplot as plt
from utils import load_material_data_train_test_split
from dataset import ndarrayDataset
from sklearn import cluster
from model.baselines import SimpleVAE
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0")
model = SimpleVAE(3600, 200, 40)
model.load_state_dict(torch.load('checkpoints/VAE_300.pth'))

data_location = "/mnt/storage/tmwang/Materials/MP.mat"
X_train,X_test,y_train,y_test,e_train, e_test = load_material_data_train_test_split(data_location, return_energy=True)
train_dataset = ndarrayDataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=True)

for (data, _)  in train_loader:
    print("Here")
    latent = model.latent_space(data).detach().numpy()

tsne = TSNE(n_components=2, perplexity=20, n_iter=1000)
tsne_results = tsne.fit_transform(latent)

plt.figure()
plt.scatter(
    x=tsne_results[:,0], y=tsne_results[:,1],c=y_train,alpha=0.5, cmap="Spectral", s = e_train+1.0
)
plt.colorbar()
plt.savefig('cluster_tsne.png')

tsne = TSNE(n_components=2, perplexity=20, n_iter=1000)
tsne_results = tsne.fit_transform(latent)

plt.figure()
plt.scatter(
    x=tsne_results[:,0], y=tsne_results[:,1], c=np.log(e_train+2.0), alpha=0.5, cmap="Spectral", s = e_train+1.0
)
plt.colorbar()
plt.savefig('cluster_tsne_energy.png')


kmeans= cluster.MiniBatchKMeans(n_clusters=7)
kmeans.fit(latent)
if hasattr(kmeans, 'labels_'):
    y_pred = kmeans.labels_.astype(np.int)
else:
    y_pred = kmeans.predict(latent)
plt.figure()
plt.scatter(
    x=tsne_results[:,0], y=tsne_results[:,1],c=y_pred,alpha=0.5, cmap="Spectral", s = e_train+1.0
)
plt.colorbar()
plt.savefig('cluster_tsne_processed.png')
