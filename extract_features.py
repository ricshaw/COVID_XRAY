import numpy as np
import matplotlib.pyplot as plt
import os
#import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage.transform import resize

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms

from efficientnet_pytorch import EfficientNet


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    #img = Image.open(path).convert('L')
    #img = np.array(img)
    #img = np.clip(img, np.percentile(img,5), np.percentile(img,95))
    #img -= img.min()
    #img /= img.max()
    #img -= np.mean(img)
    #img /= (np.std(img) + 1e-9)
    #img = Image.fromarray(np.uint8(img))
    return img

def dicom_image_loader(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img = np.uint8(255.0*img)
    img = Image.fromarray(img).convert("RGB")
    return img

class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform

    def __getitem__(self, index):
        filepath = self.df.filepath[index]
        image = self.loader(filepath)
        image = self.transform(image)
        return image, filepath

    def __len__(self):
        return self.df.shape[0]

class Model(nn.Module):
    def __init__(self, encoder='efficientnet-b3'):
        super(Model, self).__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        }
        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.net = EfficientNet.from_pretrained(encoder, num_classes=1)

    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        out = nn.Flatten()(x)
        #out = self.net(x)
        return out



filepaths = 'KCH_CXR_JPG.csv'
#filepaths = 'PUBLIC.csv'
encoder = 'efficientnet-b0'
bs = 128

df = pd.read_csv(filepaths)
print(df.shape)
print(df.head())

input_size = (512,512)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                              transforms.Resize(input_size,3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])


dataset = ImageDataset(df, transform)
data_loader = DataLoader(dataset, batch_size=bs, num_workers=8, shuffle=False)

model = Model(encoder)
use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)
#if use_cuda and torch.cuda.device_count() > 1:
#    print('Using', torch.cuda.device_count(), 'GPUs!')
#    model = nn.DataParallel(model)
model.cuda()

model.eval()
features = []
with torch.no_grad():
    for i, sample in enumerate(data_loader):
        images, names = sample[0], sample[1]
        print(i, names, images.shape)
        images = images.cuda()
        out = model(images)
        print(out.shape)
        features += out.cpu().numpy().tolist()

print('features', len(features))
features = np.stack((features))
print('features', features.shape)
np.save(encoder+'-features.npy', features)

print('END')
