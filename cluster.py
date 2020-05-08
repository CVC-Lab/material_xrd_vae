from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np 
from matplotlib import pyplot as plt
from utils import load_material_data_train_test_split
from sklearn import cluster

with open('results/GMVAE_latent.npy','rb') as f:
    latent = np.load(f)


data_location = "/mnt/storage/tmwang/Materials/MP.mat"
X_train,X_test,y_train,y_test,e_train, e_test = load_material_data_train_test_split(data_location, return_energy=True)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(latent)

plt.figure()
plt.scatter(
    x=pca_result[:,0], y=pca_result[:,1],c=y_train,alpha=0.5, cmap="Spectral", s=e_train+1.0
)
plt.colorbar()
plt.savefig('cluster_pca.png')

tsne = TSNE(n_components=2, perplexity=10, n_iter=5000)
tsne_results = tsne.fit_transform(latent)

plt.figure()
plt.scatter(
    x=tsne_results[:,0], y=tsne_results[:,1],c=y_train,alpha=0.5, cmap="Spectral", s = e_train+1.0
)
plt.colorbar()
plt.savefig('cluster_tsne.png')


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
