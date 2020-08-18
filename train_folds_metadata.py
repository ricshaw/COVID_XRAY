import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
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
encoder = 'efficientnet-b6'
EPOCHS = 30
MIN_EPOCHS = 15
PATIENCE = 5
bs = 32
input_size = (384,384)
FOLDS = 5
alpha = 0.75
gamma = 2.0
FEATURES = False
CUTMIX_PROB = 0.0
OCCLUSION = False
SAVE = True
SAVE_NAME = encoder + '-bs%d-%d-tta-ranger' % (bs, input_size[0])
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
    def __init__(self, encoder='efficientnet-b3', features=False):
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

        if features:
            n_feats = 80
            hidden1 = 256
            hidden2 = 256
            dropout = 0.3
            self.fc1 = nn.Linear(n_feats, hidden1, bias=True)
            self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
            self.meta = nn.Sequential(self.fc1,
                                      nn.ReLU(),
                                      nn.Dropout(p=dropout),
                                      self.fc2,
                                      nn.ReLU(),
                                      nn.Dropout(p=dropout)
                                     )
            self.classifier = nn.Sequential(nn.Dropout(p=params_dict[encoder][-1]),
                                            nn.Linear(in_features=n_channels_dict[encoder]+hidden2, out_features=1, bias=True)
                                            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(p=params_dict[encoder][-1]),
                                            nn.Linear(in_features=n_channels_dict[encoder], out_features=1, bias=True)
                                           )

    def forward(self, x, features=None):
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        x = nn.Flatten()(x)

        if features is not None:
            features = self.meta(features)
            x = torch.cat([x, features], dim=1)

        out = self.classifier(x)
        return out


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
                         A.Resize(input_size[0], input_size[1], interpolation=3, p=1),
                         A.ToGray(p=1),
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=3, border_mode=4, p=0.5),
                         #A.Rotate(p=1, limit=45, interpolation=3),
                         #A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=3, p=1),
                         #A.OneOf([
                         #         A.IAAAdditiveGaussianNoise(),
                         #         A.GaussNoise(),
                         #        ], p=0.0),
                         #A.OneOf([
                         #         A.MotionBlur(p=0.25),
                         #         A.MedianBlur(blur_limit=3, p=0.25),
                         #         A.Blur(blur_limit=3, p=0.25),
                         #         A.GaussianBlur(p=0.25)
                         #        ], p=0.0),
                         #A.OneOf([
                         #         A.OpticalDistortion(interpolation=3, p=0.1),
                         #         A.GridDistortion(interpolation=3, p=0.1),
                         #         A.IAAPiecewiseAffine(p=0.5),
                         #        ], p=0.0),
                         #A.OneOf([
                         #         A.CLAHE(clip_limit=2),
                         #         A.IAASharpen(),
                         #         A.IAAEmboss(),
                         #        ], p=0),
                         A.RandomBrightnessContrast(p=0.5),
                         A.RandomGamma(p=0.5),
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
val_transform = transforms.Compose([
                              transforms.Resize(input_size, 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
if False:
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
else:
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

transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),

transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
                                                                     ])),
                                         transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean, std)(image) for image in images]))
                                       ])


## Train
print('\nStarting training!')

val_preds = []
val_labels = []
val_names = []
val_auc = []

for fold in range(FOLDS):
    print('\nFOLD', fold)

    ## Init dataloaders
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    print('Train', train_df.shape)
    print('Valid', val_df.shape)

    train_dataset = ImageDataset(train_df, train_transform, A_transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4, shuffle=True, drop_last=True)

    val_dataset = ImageDataset(val_df, tta_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)

    ## Init model
    model = Model(encoder, FEATURES)
    use_cuda = torch.cuda.is_available()
    print('Using cuda', use_cuda)
    if use_cuda and torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(logits=True)
    optimizer = RangerLars(model.parameters())
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    running_auc = []
    running_preds = []
    best_auc = 0.0
    stop_count = 0

    for epoch in range(EPOCHS):

        print('\nTraining step')
        model.cuda()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        res_name = []
        res_prob = []
        res_label = []
        for i, sample in enumerate(train_loader):
            images, features, names, labels = sample[0], sample[1], sample[2], sample[3]
            #print('images', images.shape, 'features', features.shape, 'labels', labels.shape)
            images = images.cuda()
            features = features.cuda()
            labels = labels.cuda()
            labels = labels.unsqueeze(1).float()

            ## CUTMIX
            prob = np.random.rand(1)
            if prob < CUTMIX_PROB:
                # generate mixed sample
                lam = np.random.beta(1,1)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                features_a = features
                features_b = features[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                features = features_a * lam + features_b * (1. - lam)
                # compute output
                if FEATURES:
                    out = model(images, features)
                else:
                    out = model(images)
                #loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
                loss = sigmoid_focal_loss(out, target_a, alpha, gamma, reduction="mean") * lam + \
                       sigmoid_focal_loss(out, target_b, alpha, gamma, reduction="mean") * (1. - lam)

            else:
                if FEATURES:
                    out = model(images, features)
                else:
                    out = model(images)
                #loss = criterion(out, labels)
                loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            total += labels.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.5).int() == labels).sum().item()
            #print("iter: {}, Loss: {}".format(i, loss.item()) )

            res_prob += out.detach().cpu().numpy().tolist()
            res_label += labels.detach().cpu().numpy().tolist()

        # Scores
        y_true = np.array(res_label)
        y_scores = np.array(res_prob)
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        print("Epoch: {}, Loss: {}, Train Accuracy: {}, AUC: {}".format(epoch, running_loss, round(correct/total, 4), auc))

        # Tensorboard
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
        writer.add_image('images', grid, epoch)
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('AUC/train', auc, epoch)
        writer.add_scalar('AP/train', ap, epoch)

        # Save last model
        if SAVE and (epoch==(EPOCHS-1)):
            MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, epoch)))
            torch.save(model.state_dict(), MODEL_PATH)


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
                    images = images.view(batch_size, n_crops, -1)[:,0,...]
                    oc_images = images[(labels==1).squeeze()].cuda()
                    oc_labels = labels[(labels==1).squeeze()].cuda()
                count += 1

        # Scores
        acc = correct/total
        y_true = np.array(res_label)
        y_scores = np.array(res_prob)
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        running_auc.append(auc)
        running_preds.append(y_scores)
        id = int(np.argmax(running_auc))
        print('All AUCs', running_auc)
        print('Best AUC', id, running_auc[id])
        print("Epoch: {}, Loss: {}, Test Accuracy: {}, AUC: {}".format(epoch, running_loss, round(acc, 4), auc))

        # Tensorboard
        writer.add_scalar('Loss/val', loss.item(), epoch)
        writer.add_scalar('AUC/val', auc, epoch)
        writer.add_scalar('AP/val', ap, epoch)

        # Save best model so far
        if auc > best_auc:
            best_auc = auc
            stop_count = 0
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_best.pth' % (fold)))
                torch.save(model.state_dict(), MODEL_PATH)
        else:
            stop_count += 1

        print('Stop count', stop_count)
        # Early stopping
        if ((epoch>=MIN_EPOCHS) and (stop_count>PATIENCE)) or (epoch==(EPOCHS-1)):
            print('Stopping!')
            val_preds += running_preds[id].tolist()
            val_auc += [running_auc[id]]
            val_labels += res_label
            val_names += res_name
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, epoch)))
                torch.save(model.state_dict(), MODEL_PATH)
            break

        # Occlusion
        if OCCLUSION:
            oc = Occlusion(model)
            x_shape = 16
            x_stride = 8
            print('oc_images', oc_images.shape)
            print('oc_labels', oc_labels.shape)
            baseline = torch.zeros_like(oc_images).cuda()
            oc_attributions = oc.attribute(oc_images, sliding_window_shapes=(3, x_shape, x_shape),
                                        strides=(3, int(x_stride/2), int(x_stride/2)), target=0,
                                        baselines=baseline)
            oc_attributions = torch.abs(oc_attributions)
            print('oc_attributions', oc_attributions.shape)
            image_grid = torchvision.utils.make_grid(oc_images, nrow=4, normalize=True, scale_each=True)
            #oc_attributions_grid = torchvision.utils.make_grid(oc_attributions)

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

            writer.add_image('Interpretability/Image', image_normaliser(image_grid), epoch)
            writer.add_image('Interpretability/OC_Attributions_Died', image_normaliser(oc_attributions_grid), epoch)


## Totals
val_labels = np.array(val_labels)
val_preds = np.array(val_preds)
val_auc = np.array(val_auc)
print('Labels', len(val_labels), 'Preds', len(val_preds), 'AUCs', len(val_auc), 'names', len(val_names))
correct = ((val_preds > 0.5).astype(int) == val_labels).sum()
acc = correct / len(val_labels)
auc = roc_auc_score(val_labels, val_preds)
print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), auc))
print('AUC mean:', np.mean(val_auc), 'std:', np.std(val_auc))

res_prob = [x[0] for x in res_prob]
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels.tolist(), "Pred":val_preds.tolist()})
sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

## ROC curve
fpr, tpr, _ = roc_curve(val_labels, val_preds)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Prediction of Death - ROC')
plt.legend(loc="lower right")
plt.savefig('roc-' + SAVE_NAME + '.png', dpi=300)

## Prescision-Recall
average_precision = average_precision_score(val_labels, val_preds)
precision, recall, thresholds = precision_recall_curve(val_labels, val_preds)
pr_auc = metrics.auc(recall, precision)
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Prediction of Death - Precision-Recall')
plt.legend(loc="lower right")
plt.savefig('precision-recall-' + SAVE_NAME + '.png', dpi=300)

## Preds dataframe
val_labels = [x[0] for x in val_labels]
val_preds = [x[0] for x in val_preds]
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels, "Pred":val_preds})
sub.to_csv('preds-' + SAVE_NAME + '.csv', index=False)
print('END')
