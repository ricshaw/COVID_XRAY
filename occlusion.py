import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
#import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage import color
from sklearn import metrics
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

import albumentations as A
from efficientnet_pytorch import EfficientNet
from focal_loss import sigmoid_focal_loss, sigmoid_focal_loss_star
from cutout import Cutout

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


## Config
labels = '/nfs/home/richard/COVID_XRAY/cxr_news2_pseudonymised_filenames_latest_folds.csv'
encoder = 'efficientnet-b3'
EPOCHS = 100
MIN_EPOCHS = 40
PATIENCE = 5
bs = 16
input_size = (480,480)
FOLDS = 5
alpha = 0.75
gamma = 2.0
FEATURES = False
CUTOUT = False
CUTMIX_PROB = 0.0
OCCLUSION = True
SAVE = True
#efficientnet-b3-bs16-480-tta-ranger-cutmix
SAVE_NAME = encoder + '-bs%d-%d-tta-ranger-cutmix' % (bs, input_size[0])
SAVE_PATH = '/nfs/home/richard/COVID_XRAY/' + SAVE_NAME
print(SAVE_NAME)
log_name = './runs/' + SAVE_NAME
writer = SummaryWriter(log_dir=log_name)
if SAVE:
    os.makedirs(SAVE_PATH, exist_ok=True)



def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img

def dicom_image_loader(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img = np.uint8(255.0*img)
    img = Image.fromarray(img).convert("RGB")
    return img

def image_normaliser(some_image):
    return 1 * (some_image - torch.min(some_image)) / (torch.max(some_image) - torch.min(some_image))

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class ImageDataset(Dataset):
    def __init__(self, df, transform, A_transform=None):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform

    def __getitem__(self, index):
        filepath = self.df.Filename[index]
        image = self.loader(filepath)

        # Centre crop
        image_size = image.size
        small_edge = min(image_size)
        centre_crop = transforms.CenterCrop(small_edge)
        image = centre_crop(image)

        # A transform
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)

        image = self.transform(image)
        label = self.df.Died[index]

        # Features
        age = self.df.Age[index].astype(np.float32)
        gender = self.df.Gender[index].astype(np.float32)
        ethnicity = self.df.Ethnicity[index].astype(np.float32)
        onset_to_scan = df.days_from_onset_to_scan[index].astype(np.float32)
        first_blood = '.cLac'
        last_blood = 'OBS BMI Calculation'
        bloods = self.df.loc[index, first_blood:last_blood].values.astype(np.float32)
        first_vital = 'Fever (finding)'
        last_vital = 'Immunodeficiency disorder (disorder)'
        vitals = self.df.loc[index, first_vital:last_vital].values.astype(np.float32)
        features = np.concatenate((bloods, [age, gender, ethnicity, onset_to_scan], vitals), axis=0)

        return image, features, filepath, label

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
        n_feats = 80
        hidden1 = 256
        hidden2 = 256
        dropout = 0.3
        self.fc1 = nn.Linear(n_feats, hidden1, bias=True)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.meta = nn.Sequential(self.fc1,
                                  #nn.BatchNorm1d(hidden1),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  self.fc2,
                                  #nn.BatchNorm1d(hidden2),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout)
                                 )

        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=params_dict[encoder][-1]),
                                        nn.Linear(in_features=n_channels_dict[encoder], out_features=1, bias=True)
                                       )
        self.classifier_feats = nn.Sequential(nn.Dropout(p=params_dict[encoder][-1]),
                                        nn.Linear(in_features=n_channels_dict[encoder]+hidden2, out_features=1, bias=True)
                                       )
        #self.net = EfficientNet.from_pretrained(encoder, num_classes=1)

    def forward(self, x, features=None):
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        x = nn.Flatten()(x)

        if features is not None:
            features = self.meta(features)
            x = torch.cat([x, features], dim=1)
            out = self.classifier_feats(x)
        else:
            out = self.classifier(x)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


## Load labels
df = pd.read_csv(labels)
print(df.shape)
print(df.head())
print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])

## Replace data
df.Age.replace(120, np.nan, inplace=True)
df.Ethnicity.replace('Unknown', np.nan, inplace=True)
df.Ethnicity.replace('White', 1, inplace=True)
df.Ethnicity.replace('Black', 2, inplace=True)
df.Ethnicity.replace('Asian', 3, inplace=True)
df.Ethnicity.replace('Mixed', 4, inplace=True)
df.Ethnicity.replace('Other', 5, inplace=True)

# Extract features
first_blood = '.cLac'
last_blood = 'OBS BMI Calculation'
bloods = df.loc[:,first_blood:last_blood].values.astype(np.float32)
print('Bloods', bloods.shape)
first_vital = 'Fever (finding)'
last_vital = 'Immunodeficiency disorder (disorder)'
vitals = df.loc[:,first_vital:last_vital].values.astype(np.float32)
print('Vitals', vitals.shape)
age = df.Age[:,None]
gender = df.Gender[:,None]
ethnicity = df.Ethnicity[:,None]
onset_to_scan = df.days_from_onset_to_scan[:,None]

# Normalise features
scaler = StandardScaler()
X = np.concatenate((bloods, age, gender, ethnicity, onset_to_scan), axis=1)
scaler.fit(X)
X = scaler.transform(X)
X = np.concatenate((X, vitals), axis=1)
print(X.shape)

# Fill missing
print('Features before', np.nanmin(X), np.nanmax(X))
print('Missing before: %d' % sum(np.isnan(X).flatten()))
imputer = SimpleImputer(strategy='constant', fill_value=0)
imputer.fit(X)
X = imputer.transform(X)
print('Features after', np.nanmin(X), np.nanmax(X))
print('Missing after: %d' % sum(np.isnan(X).flatten()))

df.loc[:,first_blood:last_blood] = X[:,0:bloods.shape[1]]
df.loc[:,'Age'] = X[:,bloods.shape[1]]
df.loc[:,'Gender'] = X[:,bloods.shape[1]+1]
df.loc[:,'Ethnicity'] = X[:,bloods.shape[1]+2]
df.loc[:,'days_from_onset_to_scan'] = X[:,bloods.shape[1]+3]
df.loc[:,first_vital:last_vital] = X[:,bloods.shape[1]+4::]

## Transforms
A_transform = A.Compose([
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.Rotate(p=1, limit=45, interpolation=3),
                         A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=3, p=1),
                         A.OneOf([
                                  A.IAAAdditiveGaussianNoise(),
                                  A.GaussNoise(),
                                 ], p=0.3),
                         A.OneOf([
                                  A.MotionBlur(p=0.25),
                                  A.MedianBlur(blur_limit=3, p=0.25),
                                  A.Blur(blur_limit=3, p=0.25),
                                  A.GaussianBlur(p=0.25)
                                 ], p=0.1),
                         A.OneOf([
                                  A.OpticalDistortion(interpolation=3, p=0.1),
                                  A.GridDistortion(interpolation=3, p=0.1),
                                  A.IAAPiecewiseAffine(p=0.5),
                                 ], p=0.05),
                         A.OneOf([
                                  A.CLAHE(clip_limit=2),
                                  A.IAASharpen(),
                                  A.IAAEmboss(),
                                  A.RandomBrightnessContrast(),
                                 ], p=1),
                         A.RandomGamma(p=1),
                        ], p=1)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
                              #transforms.Resize(input_size, 3),
                              #transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=3),
                              #transforms.RandomHorizontalFlip(),
                              #transforms.RandomVerticalFlip(),
                              #transforms.RandomRotation(90),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
if CUTOUT:
    train_transform.transforms.append(Cutout(n_holes=1, length=160))
val_transform = transforms.Compose([
                              transforms.Resize(input_size, 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
tta_transform = transforms.Compose([transforms.Resize(input_size, 3),
                                    transforms.Lambda(lambda image: torch.stack([
                                                                     transforms.ToTensor()(image),
                                                                     transforms.ToTensor()(image.rotate(90, resample=0)),
                                                                     transforms.ToTensor()(image.rotate(180, resample=0)),
                                                                     transforms.ToTensor()(image.rotate(270, resample=0)),
                                                                     transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM)),
                                                                     transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(90, resample=0)),
                                                                     transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(180, resample=0)),
                                                                     transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(270, resample=0)),
                                                                     ])),
                                         transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean, std)(image) for image in images]))
                                       ])


print('\nStarting occlusion!')

val_preds = []
val_labels = []
val_names = []
val_auc = []

for fold in range(1):
    print('\nFOLD', fold)

    val_df = df[df.fold == fold]
    val_df.reset_index(drop=True, inplace=True)
    print('Valid', val_df.shape)

    val_dataset = ImageDataset(val_df, tta_transform)
    val_loader = DataLoader(val_dataset, batch_size=int(bs/2), num_workers=4)

    ## Init model
    model = Model(encoder)
    use_cuda = torch.cuda.is_available()
    print('Using cuda', use_cuda)
    if use_cuda and torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)

    optimizer = RangerLars(model.parameters())
    running_auc = []
    running_preds = []

    MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_best.pth' % (fold)))
    model.load_state_dict(torch.load(MODEL_PATH))
    print('Loaded', MODEL_PATH)

    if True:
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
            for images, features, names, labels in val_loader:
                images = images.cuda()
                features = features.cuda()
                labels = labels.cuda()
                labels = labels.unsqueeze(1).float()

                ## TTA
                batch_size, n_crops, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                _, n_feats = features.size()
                features = features.repeat(1,n_crops).view(-1,n_feats)
                if FEATURES:
                    out = model(images, features)
                else:
                    out = model(images)
                out = out.view(batch_size, n_crops, -1).mean(1)
                print(images.shape, out.shape)

                #loss = criterion(out.data, labels)
                loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

                running_loss += loss.item()

                total += labels.size(0)
                out = torch.sigmoid(out)
                correct += ((out > 0.5).int() == labels).sum().item()

                res_prob += out.cpu().numpy().tolist()
                res_label += labels.cpu().numpy().tolist()
                res_name += names

                if OCCLUSION and (count==0):
                    labels = labels.repeat(1,n_crops).view(-1,1)
                    print('occ1', images.shape, labels.shape)
                    #images = images.view(batch_size, n_crops, c, h, w)
                    #print('occ2', images.shape)
                    #images = images[:,0,...]
                    #print('occ3', images.shape)
                    oc_images = images[(labels==1).squeeze()].cuda()
                    oc_labels = labels[(labels==1).squeeze()].cuda()
                count += 1

        # Occlusion
        if OCCLUSION:
            oc = Occlusion(model)
            x_shape = 128
            x_stride =  64
            print('oc_images', oc_images.shape)
            print('oc_labels', oc_labels.shape)
            baseline = torch.zeros_like(oc_images).cuda()
            oc_attributions = oc.attribute(oc_images, sliding_window_shapes=(3, x_shape, x_shape),
                                        strides=(3, int(x_stride/2), int(x_stride/2)), target=0,
                                        baselines=baseline)
            oc_attributions = torch.abs(oc_attributions)
            print('oc_attributions', oc_attributions.shape)
            image_grid = torchvision.utils.make_grid(oc_images, nrow=8, normalize=True, scale_each=True)
            image_grid = 255.0*image_grid.cpu().numpy().transpose(1,2,0)
            print(image_grid.shape)
            cv2.imwrite('image_grid.png',image_grid)

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
            oc_attributions_grid = torchvision.utils.make_grid(oc_attributions_grid, nrow=8, normalize=True, scale_each=True)
            print(oc_attributions_grid.shape)
            oc_attributions_grid = 255.0*oc_attributions_grid.cpu().numpy().transpose(1,2,0)
            cv2.imwrite('im_grid.png', oc_attributions_grid[...,::-1])

            def colmap(im):
                im = color.rgb2gray(im)
                im -= im.min()
                im /= im.max()
                #im = cmapper(im)[...,:3]
                im = Image.fromarray(np.uint8(255*im))
                im = transforms.ToTensor()(im)
                return im

            n=8
            oc_attributions = oc_images.cpu().numpy()
            oc_attributions_mean = []
            oc_attributions_std = []
            for i in range(2):
                im0 = np.array(oc_attributions[n*i,...]).transpose([1,2,0])
                im1 = np.array(oc_attributions[n*i+1,...]).transpose([1,2,0])
                im2 = np.array(oc_attributions[n*i+2,...]).transpose([1,2,0])
                im3 = np.array(oc_attributions[n*i+3,...]).transpose([1,2,0])
                im4 = np.array(oc_attributions[n*i+4,...]).transpose([1,2,0])
                im5 = np.array(oc_attributions[n*i+5,...]).transpose([1,2,0])
                im6 = np.array(oc_attributions[n*i+6,...]).transpose([1,2,0])
                im7 = np.array(oc_attributions[n*i+7,...]).transpose([1,2,0])
                im1 = np.rot90(im1,k=-1)
                im2 = np.rot90(im2,k=-2)
                im3 = np.rot90(im3,k=-3)
                #print((np.abs(im0-im1)).sum(), im0.min(), im1.min(), im0.max(), im1.max())
                #print((np.abs(im0-im2)).sum(), im0.min(), im2.min(), im0.max(), im2.max())
                #print((np.abs(im0-im3)).sum(), im0.min(), im3.min(), im0.max(), im3.max())
                im4 = np.flipud(im4)
                im5 = np.rot90(np.flipud(im5),k=-3)
                im6 = np.rot90(np.flipud(im6),k=-2)
                im7 = np.rot90(np.flipud(im7),k=-1)
                #print((np.abs(im0-im4)).sum(), im0.min(), im4.min(), im0.max(), im4.max())
                #print((np.abs(im0-im5)).sum(), im0.min(), im5.min(), im0.max(), im5.max())
                #print((np.abs(im0-im6)).sum(), im0.min(), im6.min(), im0.max(), im6.max())
                #print((np.abs(im0-im7)).sum(), im0.min(), im7.min(), im0.max(), im7.max())
                im = np.stack((im0,im1,im2,im3,im4,im5,im6,im7), axis=-1)
                im_mean = np.mean(im, axis=-1)
                im_std = np.std(im, axis=-1)
                print(im_mean.shape, im_std.shape)
                print('std', im_std.min(), im_std.max(), im_std.sum())
                im_mean = colmap(im_mean)
                im_std = colmap(im_std)
                oc_attributions_mean.append(im_mean)
                oc_attributions_std.append(im_std)
            oc_attributions_mean = torchvision.utils.make_grid(oc_attributions_mean, nrow=2, normalize=True, scale_each=True)
            oc_attributions_std = torchvision.utils.make_grid(oc_attributions_std, nrow=2, normalize=True, scale_each=True)
            print(oc_attributions_mean.shape, oc_attributions_std.shape)
            oc_attributions_mean = 255.0*oc_attributions_mean.cpu().numpy().transpose(1,2,0)
            oc_attributions_std = 255.0*oc_attributions_std.cpu().numpy().transpose(1,2,0)
            cv2.imwrite('im_mean.png', oc_attributions_mean[...,::-1])
            cv2.imwrite('im_std.png', oc_attributions_std[...,::-1])

