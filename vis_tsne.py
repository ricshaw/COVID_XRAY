import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from skimage.transform import resize
import cv2

def plot_mnist(X, X_embedded, data, images=True, min_dist=64.0, savename=None):

    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker=".")

    if images:
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imgpath = data.filepath[i]
            img = cv2.imread(imgpath, 0)
            #img = np.clip(img, np.percentile(img,2), np.percentile(img,98))
            #img -= img.min()
            #img /= img.max()
            img = cv2.resize(img, (128,128))
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap='gray'), X_embedded[i])
            ax.add_artist(imagebox)
    plt.savefig(savename, dpi=400)


# Load file list
data = pd.read_csv('KCH_CXR_JPG.csv')
print(data.shape)

# Load features
X = np.load('efficientnet-b0-features.npy')
print(X.shape)

# Load embedding
X_embedded = np.load('tsne.npy')
print(X_embedded.shape)

# Plot
plot_mnist(X, X_embedded, data, images=True, min_dist=20.0, savename='tsne_images.png')
