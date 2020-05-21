from utils import load_material_data_v2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
import numpy as np

data_location = "/mnt/storage/tmwang/Materials/MP_v2.mat"
X,_,_,energy = load_material_data_v2(data_location)

# Intensity Image
intensity_mean =np.zeros([36,100])
for i in range(X.shape[0]):
    intensity_mean += np.reshape(X[i,:],[100,36]).T
intensity_mean = intensity_mean / X.shape[0]

intensity = np.reshape(X[0,:],[100,36]).T


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

#ax = plt.gca()
im = plt.imshow(intensity_mean)
im.set_cmap('nipy_spectral')
add_colorbar(im)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = plt.colorbar(im, cax=cax)
#cbar.set_ticks([-4,4])
plt.axis('off')
plt.savefig('results/intensity_array_average.png', bbox_inches = 'tight', pad_inches = 0, dpi=1000)
plt.savefig('results/intensity_array_average.svg', bbox_inches = 'tight', pad_inches = 0)

plt.clf()
im2 = plt.imshow(intensity)
im2.set_cmap('nipy_spectral')
plt.colorbar()

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.

plt.axis('off')
plt.savefig('results/intensity_array.png', bbox_inches = 'tight', pad_inches = 0)
plt.savefig('results/intensity_array.svg', bbox_inches = 'tight', pad_inches = 0)

# spectral overlap
plt.clf()
x = np.linspace(0,180,num=100,endpoint=False)
for i in range(36):
    plt.plot(x, intensity_mean[i,:]+i*0.1)
plt.savefig('results/intensity_lines.png', bbox_inches = 'tight', pad_inches = 0)
plt.savefig('results/intensity_lines.svg', bbox_inches = 'tight', pad_inches = 0)


x = np.linspace(0,180,num=100,endpoint=False)
X_sampled= X[::100,:]
for i in range(36):
    print("Generating Row %d" % (i+1))
    plt.clf()
    plt.title("Row %d" % (i+1))
    plt.xlabel("Angle")
    plt.ylabel("Intensity")
    for j in range(X_sampled.shape[0]):
        plt.plot(x, X_sampled[j,i*100:(i+1)*100]+j*0.1)   
    plt.savefig('results/intensity_lines/intensity_%d.png' % (i+1), bbox_inches = 'tight', pad_inches = 0)
    plt.savefig('results/intensity_lines/intensity_%d.svg' % (i+1), bbox_inches = 'tight', pad_inches = 0)


plt.clf()
im = plt.imshow(np.tanh(X), interpolation = 'nearest', aspect = 'auto')
im.set_cmap('nipy_spectral')
plt.colorbar()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = plt.colorbar(im, cax=cax)
#cbar.set_ticks([-4,4])
plt.axis('off')
plt.savefig('results/intensity_array_X.png', bbox_inches = 'tight', pad_inches = 0, dpi=1000)
plt.savefig('results/intensity_array_X.svg', bbox_inches = 'tight', pad_inches = 0)

# data clustering
yield_cluster=False
if yield_cluster:
    from sklearn.manifold import TSNE
    idx = (energy<10)
    X=X[idx,:]
    tsne = TSNE(n_components=2, perplexity=20, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    plt.clf()
    plt.scatter(
        x=tsne_results[:,0], y=tsne_results[:,1], c=energy, alpha=0.5, cmap="Spectral",
    )
    plt.axis('off')
    plt.colorbar()
    plt.savefig('results/cluster_XRD.svg', bbox_inches = 'tight', pad_inches = 0)