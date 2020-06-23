import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from skimage.transform import resize
import cv2

# Load features
X = np.load('efficientnet-b0-features.npy')
print(X.shape)

# tSNE embedding
X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X)
print(X_embedded.shape)
np.save('tsne.npy', X_embedded)
