import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
from PIL import Image
from PIL.Image import fromarray
from skimage import color
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

#import albumentations as A
from efficientnet_pytorch import EfficientNet
from focal_loss import sigmoid_focal_loss, sigmoid_focal_loss_star

import sys
sys.path.append('/nfs/home/richard/over9000')
from over9000 import RangerLars

from torch.utils.tensorboard import SummaryWriter

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion
)


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img


class ImageDataset(Dataset):
    def __init__(self, df, transform, A_transform=None):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform

    def __getitem__(self, index):
        filepath = self.df.Filename[index]
        image = self.loader(filepath)

        # A transform
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)

        image = self.transform(image)
        label = self.df.Died[index]
        return image, filepath, label

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
        #self.net = EfficientNet.from_pretrained(encoder)
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.classifier = nn.Sequential(nn.Flatten(),
        #                                nn.Dropout(0.5),
        #                                nn.Linear(in_features=n_channels_dict[encoder], out_features=1, bias=True)
        #                                )
        self.net = EfficientNet.from_pretrained(encoder, num_classes=1)

    def forward(self, x):
        #x = self.net.extract_features(x)
        #x = self.avg_pool(x)
        #out = self.classifier(x)
        out = self.net(x)
        return out


## Config
labels = '/nfs/home/richard/COVID_XRAY/folds.csv'
encoder = 'efficientnet-b3'
EPOCHS = 75
bs = 32
input_size = (320,320)
FOLDS = 5
alpha = 0.75
gamma = 2.0
OCCLUSION = True
SAVE_NAME = encoder + '-bs%d-%d-tta-ranger' % (bs, input_size[0])
SAVE_PATH = '/nfs/home/richard/COVID_XRAY/' + SAVE_NAME

## Load labels
df = pd.read_csv(labels)
print(df.shape)
print(df.head())
print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
                              transforms.Resize(input_size, 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])


## Occlusion
#for fold in range(FOLDS):
for fold in range(1):
    print('\nFOLD', fold)

    ## Init dataloaders
    val_df = df[df.fold == fold]
    val_df.reset_index(drop=True, inplace=True)
    print('Valid', val_df.shape)
    val_dataset = ImageDataset(val_df, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=8)

    ## Init model
    model = Model(encoder)
    use_cuda = torch.cuda.is_available()
    print('Using cuda', use_cuda)
    if use_cuda and torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters())

    ## Load model
    MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, EPOCHS-1)))
    model.load_state_dict(torch.load(MODEL_PATH))
    print('Loaded', MODEL_PATH)
    model.cuda()

    print('\nValidation step')
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    res_name = []
    res_prob = []
    res_label = []
    count = 0
    with torch.no_grad():
        for images, names, labels in val_loader:
            print(images.shape)
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.unsqueeze(1).float()

            out = model(images)

            #loss = criterion(out.data, labels)
            loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

            running_loss += loss.item()

            total += labels.size(0)
            out = torch.sigmoid(out)
            #correct += ((out > 0.5).int() == labels).sum().item()

            #res_prob += out.cpu().numpy().tolist()
            #res_label += labels.cpu().numpy().tolist()

            #val_preds += out.cpu().numpy().tolist()
            #val_labels += labels.cpu().numpy().tolist()
            #val_names += names

            if count == 0:
                oc_images = images[(labels==1).squeeze()].cuda()
                oc_labels = labels[(labels==1).squeeze()].cuda()
            count += 1


    ## Occlusion
    if OCCLUSION:
        print('Computing occlusion')
        oc = Occlusion(model)
        x_shape = 32 #16
        x_stride = 16 #8
        print('oc_images', oc_images.shape)
        print('oc_labels', oc_labels.shape)
        baseline = torch.zeros_like(oc_images).cuda()
        oc_attributions = oc.attribute(oc_images, sliding_window_shapes=(3, x_shape, x_shape),
                                        strides=(3, int(x_stride/2), int(x_stride/2)), target=0,
                                        baselines=baseline)
        oc_attributions = torch.abs(oc_attributions)
        print('oc_attributions', oc_attributions.shape)
        image_grid = torchvision.utils.make_grid(oc_images, nrow=4, normalize=True, scale_each=True)
        image_grid = image_grid.cpu().numpy()
        cv2.imwrite('image_grid.png', image_grid)

        oc_attributions = oc_attributions.cpu().numpy()
        cmapper = matplotlib.cm.get_cmap('hot')
        oc_attributions_grid = []
        for i in range(oc_attributions.shape[0]):
            im = np.array(oc_attributions[i,...])
            im = np.transpose(im,[1,2,0])
            im = color.rgb2gray(im)
            im -= im.min()
            im /= im.max()
            im = cmapper(im)[...,:3]
            im = Image.fromarray(np.uint8(255*im))
            im = transforms.ToTensor()(im)
            oc_attributions_grid.append(im)
        oc_attributions_grid = torchvision.utils.make_grid(oc_attributions_grid, nrow=4, normalize=True, scale_each=True)
        print(oc_attributions_grid.shape)
        oc_attributions_grid = oc_attributions_grid.cpu().numpy()
        cv2.imwrite('oc_attributions_grid.png', oc_attributions_grid)


