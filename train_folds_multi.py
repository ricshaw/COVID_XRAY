import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2
import sys
import random
#import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage import color
from sklearn import linear_model
from sklearn import metrics
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer, required
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms, models

import albumentations as A
from efficientnet_pytorch import EfficientNet
from focal_loss import sigmoid_focal_loss, sigmoid_focal_loss_star

sys.path.append('/nfs/home/richard/over9000')
from over9000 import RangerLars

#from captum.attr import (
#    GradientShap,
#    DeepLift,
#    DeepLiftShap,
#    IntegratedGradients,
#    LayerConductance,
#    NeuronConductance,
#    NoiseTunnel,
#    Occlusion)

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seeded!')
seed_everything(42)


## Config
data = '/nfs/home/richard/COVID_XRAY/cxr_news2_pseudonymised_filenames_folds.csv'
#test_data = '/nfs/home/richard/COVID_XRAY/gstt.csv'
test_data = '/nfs/home/richard/COVID_XRAY/gstt_new.csv'
encoder = 'efficientnet-b0'
IMAGES = False
FEATS = True
MULTI = True
EPOCHS = 50
MIN_EPOCHS = 50
PATIENCE = 5
bs = 64
input_size = (256,256)
FOLDS = 5
#alpha = 0.75
gamma = 2.0
CUTMIX_PROB = 0.0
OCCLUSION = False
SAVE = True
SAVE_NAME = 'multi_' + encoder + '-bs%d-%d' % (bs, input_size[0])
if IMAGES:
    SAVE_NAME += '_images'
if FEATS:
    SAVE_NAME += '_feats'
SAVE_PATH = '/nfs/home/richard/COVID_XRAY/' + SAVE_NAME
print(SAVE_NAME)
log_name = './runs/' + SAVE_NAME
writer = SummaryWriter(log_dir=log_name)
if SAVE:
    os.makedirs(SAVE_PATH, exist_ok=True)


## Load train data
df = pd.read_csv(data)
print('Train data:', df.shape)
print(df.head())
print('Number of images:', df.shape[0])
print('Died:', df[df.Died == 1].shape[0])
print('Survived:', df[df.Died == 0].shape[0])

## Load test data
test_df = pd.read_csv(test_data)
print('Test data:', test_df.shape)
tmp = test_df.drop(columns=['patient_pseudo_id','CXR_datetime','Age','Gender','Ethnicity','Died'])
bloods_cols = tmp.columns
print('Bloods:', bloods_cols)


def new_section(text):
    print('\n')
    print(100 * '*')
    print(text)

def prepare_data(df, bloods_cols):
    print('Preparing data')
    # Define columns
    #first_blood = '.cLac'
    #last_blood = 'BMI'
    #first_vital = 'Fever (finding)'
    #last_vital = 'Immunodeficiency disorder (disorder)'

    ## Replace data
    df.Age.replace(120, np.nan, inplace=True)
    df.Ethnicity.replace('Unknown', np.nan, inplace=True)
    df.Ethnicity.replace('White', 1, inplace=True)
    df.Ethnicity.replace('Black', 2, inplace=True)
    df.Ethnicity.replace('Asian', 3, inplace=True)
    df.Ethnicity.replace('Mixed', 4, inplace=True)
    df.Ethnicity.replace('Other', 5, inplace=True)

    # Compute relative timestamps
    df_tmp = df.pop('CXR_datetime')
    df['CXR_datetime'] = df_tmp
    df['CXR_datetime'] = pd.to_datetime(df.CXR_datetime, dayfirst=True)
    df['min_datetime'] = df['CXR_datetime']
    df['min_datetime'] = df.groupby('patient_pseudo_id')['min_datetime'].transform('min')
    df['rel_datetime'] = (df['CXR_datetime'] - df['min_datetime']) / np.timedelta64(1,'D')
    #print(df.head(10))

    # Extract features
    #bloods = df.loc[:,first_blood:last_blood].values.astype(np.float32)
    bloods = df.loc[:,bloods_cols].values.astype(np.float32)
    print('Bloods', bloods.shape)
    #vitals = df.loc[:,first_vital:last_vital].values.astype(np.float32)
    #print('Vitals', vitals.shape)
    age = df.Age[:,None]
    gender = df.Gender[:,None]
    ethnicity = df.Ethnicity[:,None]
    time = df.rel_datetime[:,None]

    # Normalise features
    scaler = StandardScaler()
    X = np.concatenate((bloods, age, gender, ethnicity, time), axis=1)
    scaler.fit(X)
    X = scaler.transform(X)
    #X = np.concatenate((X, vitals), axis=1)
    #print(X.shape)

    # Fill missing
    #print('Features before', np.nanmin(X), np.nanmax(X))
    #print('Missing before: %d' % sum(np.isnan(X).flatten()))
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(X)
    X = imputer.transform(X)
    #print('Features after', np.nanmin(X), np.nanmax(X))
    #print('Missing after: %d' % sum(np.isnan(X).flatten()))

    # Put back features
    #df.loc[:,first_blood:last_blood] = X[:,0:bloods.shape[1]]
    df.loc[:,bloods_cols] = X[:,0:bloods.shape[1]]
    df.loc[:,'Age'] = X[:,bloods.shape[1]]
    df.loc[:,'Gender'] = X[:,bloods.shape[1]+1]
    df.loc[:,'Ethnicity'] = X[:,bloods.shape[1]+2]
    df.loc[:,'rel_datetime'] = X[:,bloods.shape[1]+3]
    #df.loc[:,first_vital:last_vital] = X[:,bloods.shape[1]+3::]
    return df

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

def get_feats(df, i, aug=False):
    age = df.Age[i].astype(np.float32)
    gender = df.Gender[i].astype(np.float32)
    ethnicity = df.Ethnicity[i].astype(np.float32)
    #first_blood = '.cLac'
    #last_blood = 'BMI'
    bloods = df.loc[i, bloods_cols].values.astype(np.float32)
    if aug:
        bloods += np.random.normal(0, 0.2, bloods.shape)
    #first_vital = 'Fever (finding)'
    #last_vital = 'Immunodeficiency disorder (disorder)'
    #vitals = df.loc[i, first_vital:last_vital].values.astype(np.float32)
    #feats = np.concatenate((bloods, [age, gender, ethnicity], vitals), axis=0)
    feats = np.concatenate((bloods, [age, gender, ethnicity]), axis=0)
    return feats

class ImageDataset(Dataset):
    def __init__(self, my_df, transform, A_transform=None):
        self.df = my_df
        self.unique_df = self.df.drop_duplicates(subset='patient_pseudo_id').reset_index(drop=True)
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform

    def __getitem__(self, index):
        image1, image2 = np.array([]), np.array([])
        feats1, feats2 = np.array([]), np.array([])

        # Select unique patient
        pid = self.unique_df.patient_pseudo_id[index]
        # Get all patient data
        pid_df = self.df[self.df['patient_pseudo_id']==pid].reset_index(drop=True)
        # Randomly choose two data points with replacement
        pid_df = pid_df.loc[np.random.choice(pid_df.shape[0], 2, replace=True)].reset_index(drop=True)
        # Sort two samples by time
        pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
        pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
        # Times
        time1 = pid_df.rel_datetime[0].astype(np.float32)
        time2 = pid_df.rel_datetime[1].astype(np.float32)
        # Features
        if FEATS:
            feats1 = get_feats(pid_df, 0, aug=True)
            feats2 = get_feats(pid_df, 1, aug=True)
            feats1 = np.append(feats1, time1)
            feats2 = np.append(feats2, time2)
        # Image
        if IMAGES:
            image1 = self.loader(pid_df.Filename[0])
            image2 = self.loader(pid_df.Filename[1])
            # Centre crop
            image1 = transforms.CenterCrop(min(image1.size))(image1)
            image2 = transforms.CenterCrop(min(image2.size))(image2)
            # A transform
            if self.A_transform is not None:
                image1 = np.array(image1)
                image2 = np.array(image2)
                image1 = self.A_transform(image=image1)['image']
                image2 = self.A_transform(image=image2)['image']
                image1 = Image.fromarray(image1)
                image2 = Image.fromarray(image2)
            # Transform
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        # Label
        label = pid_df.Died[0]

        return pid, image1, image2, feats1, feats2, time1, time2, label

    def __len__(self):
        return self.unique_df.shape[0]


sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class SingleModel(nn.Module):
    def __init__(self, encoder='efficientnet-b0', nfeats=31):
        super(SingleModel, self).__init__()
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
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),}
        self.out_chns = 0
        if IMAGES:
            self.net = EfficientNet.from_pretrained(encoder)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.out_chns += n_channels_dict[encoder]
        if FEATS:
            hidden1 = 128
            hidden2 = 128
            self.out_chns += hidden2
            self.fc1 = nn.Linear(nfeats, hidden1, bias=False)
            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.meta = nn.Sequential(self.fc1,
                                      nn.BatchNorm1d(hidden1),
                                      #nn.ReLU(),
                                      Swish_Module(),
                                      nn.Dropout(p=0.5),
                                      self.fc2,
                                      nn.BatchNorm1d(hidden2),)

    def forward(self, image=None, feats=None):
        x1 = torch.FloatTensor().cuda()
        x2 = torch.FloatTensor().cuda()
        if image.nelement() is not 0:
            x1 = self.net.extract_features(image)
            x1 = self.avg_pool(x1)
            x1 = nn.Flatten()(x1)
        if feats.nelement() is not 0:
            x2 = self.meta(feats)
        x = torch.cat([x1, x2], dim=1)
        return x

class CombinedModel(nn.Module):
    def __init__(self, singlemodel):
        super(CombinedModel, self).__init__()
        self.singlemodel = singlemodel
        nchns = self.singlemodel.out_chns
        if MULTI:
            nchns *= 2
        hidden1 = 128
        self.classifier = nn.Sequential( #nn.ReLU(),
                                        Swish_Module(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(nchns, hidden1),
                                        #nn.ReLU(),
                                        Swish_Module(),)
        self.fc = nn.Linear(hidden1, 1)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def forward(self, image1=None, feats1=None, image2=None, feats2=None):
        x1 = torch.FloatTensor().cuda()
        x2 = torch.FloatTensor().cuda()
        x1 = self.singlemodel(image1, feats1)
        if MULTI:
            x2 = self.singlemodel(image2, feats2)
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out


## Transforms
A_transform = A.Compose([
                         A.Resize(input_size[0], input_size[1], interpolation=3, p=1),
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=3, border_mode=4, p=0.5),
                         #A.Rotate(p=1, limit=45, interpolation=3),
                         #A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=3, p=1),
                         #A.OneOf([
                         #         A.IAAAdditiveGaussianNoise(),
                         #         A.GaussNoise(),
                         #        ], p=0.0),
                         A.OneOf([
                                  A.MotionBlur(p=0.25),
                                  A.MedianBlur(blur_limit=3, p=0.25),
                                  A.Blur(blur_limit=3, p=0.25),
                                  A.GaussianBlur(p=0.25)
                                 ], p=0.2),
                         #A.OneOf([
                         #         A.OpticalDistortion(interpolation=3, p=0.1),
                         #         A.GridDistortion(interpolation=3, p=0.1),
                         #         A.IAAPiecewiseAffine(p=0.5),
                         #        ], p=0.0),
                         A.OneOf([
                                  A.CLAHE(clip_limit=2),
                                  A.IAASharpen(),
                                  A.IAAEmboss(),
                                 ], p=0.2),
                         A.RandomBrightnessContrast(p=0.5),
                         A.RandomGamma(p=0.5),
                         A.ToGray(p=1),
                         A.InvertImg(p=0.2)
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
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
                                                                     ])),
                                         transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean, std)(image) for image in images]))
                                       ])


## Train
new_section('Starting Training!')

## Check train dataloader
if False:
    check_dataset = ImageDataset(df, train_transform, A_transform)
    check_loader = DataLoader(check_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
    for i, sample in enumerate(check_loader):
        pid = sample[0]
        image1, image2 = sample[1], sample[2]
        feats1, feats2 = sample[3], sample[4]
        time1, time2 = sample[5], sample[6]
        labels = sample[7]
        print('\n',pid)
        print('1:', image1.shape, feats1.shape, time1)
        print('2:', image2.shape, feats2.shape, time2)
        time1 = time1.cpu().numpy()
        time2 = time2.cpu().numpy()
        image1 = image1.cpu().numpy()[0]
        image2 = image2.cpu().numpy()[0]
        image1 = np.transpose(image1, [1,2,0])
        image2 = np.transpose(image2, [1,2,0])
        image1 = 255*(image1 - image1.min()) / (image1.max() - image1.min())
        image2 = 255*(image2 - image2.min()) / (image2.max() - image2.min())
        out_image = np.concatenate((image1, image2), axis=1)
        out_name = str(pid[0]) + '_' + str(time1[0]) + '_' + str(time2[0]) + '_.jpg'
        print(out_name)
        cv2.imwrite(out_name, out_image)
    exit(0)

## Init
val_preds = []
val_labels = []
val_names = []
val_auc = []

## Fold loop
for fold in range(FOLDS):
    new_section('Fold: %d' % fold)

    ## Init dataloaders
    train_df = df[df.fold != fold].reset_index(drop=True, inplace=False)
    val_df = df[df.fold == fold].reset_index(drop=True, inplace=False)

    ## Prepare data
    train_df = prepare_data(train_df, bloods_cols).reset_index(drop=True, inplace=False)
    val_df = prepare_data(val_df, bloods_cols).reset_index(drop=True, inplace=False)

    print('Train', train_df.shape)
    print('Valid', val_df.shape)

    ## Train dataset
    train_dataset = ImageDataset(train_df, train_transform, A_transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4, shuffle=True, drop_last=True)

    ## Val dataser
    val_dataset = ImageDataset(val_df, tta_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)

    ## Init model
    singlemodel = SingleModel(encoder).cuda()
    model = CombinedModel(singlemodel).cuda()
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters())

    alpha = train_df[train_df.Died==0].shape[0]/train_df.shape[0]
    print('Alpha', alpha)

    running_auc = []
    running_preds = []
    best_auc = 0.0
    stop_count = 0

    ## Training loop
    for epoch in range(EPOCHS):

        # Training
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

            pid = sample[0]
            image1, image2 = sample[1].cuda(), sample[2].cuda()
            feats1, feats2 = sample[3].cuda(), sample[4].cuda()
            time1, time2 = sample[5], sample[6]
            labels = sample[7].cuda()
            labels = labels.unsqueeze(1).float()
            #print('image1', image1, 'feats1', feats1, 'image2', image2, 'feats2', feats2)

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
                out = model(image1, feats1, image2, feats2)
                #loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
                loss = sigmoid_focal_loss(out, target_a, alpha, gamma, reduction="mean") * lam + \
                       sigmoid_focal_loss(out, target_b, alpha, gamma, reduction="mean") * (1. - lam)

            else:
                out = model(image1, feats1, image2, feats2)
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
        #grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
        #writer.add_image('images', grid, epoch)
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('AUC/train', auc, epoch)
        writer.add_scalar('AP/train', ap, epoch)

        # Save last model
        if SAVE and (epoch==(EPOCHS-1)):
            MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, epoch)))
            torch.save(model.state_dict(), MODEL_PATH)

        ## Validation
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
            for pid in val_df['patient_pseudo_id'].unique():
                #print(pid)
                pid_df = val_df[val_df['patient_pseudo_id']==pid].reset_index(drop=True)
                pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                n_images = pid_df.shape[0]
                #print(n_images, 'images')
                ind1=0
                ind2=n_images-1
                time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                time2 = pid_df.rel_datetime[ind2].astype(np.float32)
                image1, image2 = np.array([]), np.array([])
                feats1, feats2 = np.array([]), np.array([])
                if FEATS:
                    # Features
                    feats1 = get_feats(pid_df, ind1, aug=False)
                    feats2 = get_feats(pid_df, ind2, aug=False)
                    feats1 = np.append(feats1, time1)
                    feats2 = np.append(feats2, time2)
                else:
                    feats1, feats2 = np.array([]), np.array([])
                if IMAGES:
                    # Image
                    image1 = default_image_loader(pid_df.Filename[ind1])
                    image2 = default_image_loader(pid_df.Filename[ind2])
                    # Centre crop
                    image1 = transforms.CenterCrop(min(image1.size))(image1)
                    image2 = transforms.CenterCrop(min(image2.size))(image2)
                    # Transform
                    image1 = tta_transform(image1)
                    image2 = tta_transform(image2)
                else:
                    image1, image2 = torch.FloatTensor(), torch.FloatTensor()
                # Label
                labels = pid_df.Died[ind1]
                labels = torch.Tensor([labels]).cuda()

                image1, image2 = image1.cuda(), image2.cuda()
                image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
                feats1, feats2 = torch.Tensor(feats1).cuda(), torch.Tensor(feats2).cuda()
                feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)
                labels = labels.unsqueeze(1).float()
                #print(image1.shape, image2.shape, feats1.shape, feats2.shape, labels.shape)

                ## TTA
                if len(image1.size())==5:
                     batch_size, n_crops, c, h, w = image1.size()
                     image1 = image1.view(-1, c, h, w)
                     image2 = image2.view(-1, c, h, w)
                     if FEATS:
                         _, n_feats = feats1.size()
                         feats1 = feats1.repeat(1,n_crops).view(-1,n_feats)
                         feats2 = feats2.repeat(1,n_crops).view(-1,n_feats)
                     out = model(image1, feats1, image2, feats2)
                     out = out.view(batch_size, n_crops, -1).mean(1)
                else:
                     out = model(image1, feats1, image2, feats2)

                #loss = criterion(out.data, labels)
                loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")
                running_loss += loss.item()

                total += labels.size(0)
                out = torch.sigmoid(out)
                correct += ((out > 0.5).int() == labels).sum().item()

                res_prob += out.cpu().numpy().tolist()
                res_label += labels.cpu().numpy().tolist()
                res_name += [pid]

                #if OCCLUSION and (count==0):
                #    images = images.view(batch_size, n_crops, -1)[:,0,...]
                #    oc_images = images[(labels==1).squeeze()].cuda()
                #    oc_labels = labels[(labels==1).squeeze()].cuda()
                #count += 1

        # Scores
        acc = correct/total
        y_true = np.array(res_label)
        y_scores = np.array(res_prob)
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        running_auc.append(auc)
        running_preds.append(y_scores)
        id = int(np.argmax(running_auc))
        print("Epoch: {}, Loss: {}, Test Accuracy: {}, AUC: {}".format(epoch, running_loss, round(acc, 4), auc))
        print('All AUC:', running_auc)
        print('Best AUC:', id, running_auc[id])

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
print('Labels:', len(val_labels), 'Preds:', len(val_preds), 'Names:', len(val_names), 'AUCs:', len(val_auc))
correct = ((val_preds > 0.5).astype(int) == val_labels).sum()
acc = correct / len(val_labels)
auc = roc_auc_score(val_labels, val_preds)
print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), auc))
print('AUC mean:', np.mean(val_auc), 'std:', np.std(val_auc))

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
plt.savefig(os.path.join(SAVE_PATH,'roc-' + SAVE_NAME + '.png'), dpi=300)

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
plt.savefig(os.path.join(SAVE_PATH,'precision-recall-' + SAVE_NAME + '.png'), dpi=300)

## Preds dataframe
val_labels = [x[0] for x in val_labels]
val_preds = [x[0] for x in val_preds]
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels, "Pred":val_preds})
sub.to_csv(os.path.join(SAVE_PATH,'preds-' + SAVE_NAME + '.csv'), index=False)



## Test
new_section('Testing!')
test_df = prepare_data(test_df, bloods_cols)
print('Test data:', test_df.shape)

y_pred = 0
test_accs = []
test_aucs = []
for fold in range(FOLDS):
    print('\nFOLD', fold)

    ## Load best model!
    MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_best.pth' % (fold)))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print('Loaded:', MODEL_PATH)

    res_name, res_prob, res_label  = [], [], []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for pid in test_df['patient_pseudo_id'].unique():
            #print(pid)
            pid_df = test_df[test_df['patient_pseudo_id']==pid].reset_index(drop=True)
            pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
            pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
            n_images = pid_df.shape[0]
            #print(n_images, 'images')
            ind1 = 0
            ind2 = n_images-1
            time1 = pid_df.rel_datetime[ind1].astype(np.float32)
            time2 = pid_df.rel_datetime[ind2].astype(np.float32)
            image1, image2 = np.array([]), np.array([])
            feats1, feats2 = np.array([]), np.array([])
            if FEATS:
                # Features
                feats1 = get_feats(pid_df, ind1, aug=False)
                feats2 = get_feats(pid_df, ind2, aug=False)
                feats1 = np.append(feats1, time1)
                feats2 = np.append(feats2, time2)
            else:
                feats1, feats2 = np.array([]), np.array([])
            if IMAGES:
                # Image
                image1 = default_image_loader(pid_df.Filename[ind1])
                image2 = default_image_loader(pid_df.Filename[ind2])
                # Centre crop
                image1 = transforms.CenterCrop(min(image1.size))(image1)
                image2 = transforms.CenterCrop(min(image2.size))(image2)
                # Transform
                image1 = tta_transform(image1)
                image2 = tta_transform(image2)
            else:
                image1, image2 = torch.FloatTensor(), torch.FloatTensor()
            # Label
            labels = pid_df.Died[ind1]
            labels = torch.Tensor([labels]).cuda()

            image1, image2 = image1.cuda(), image2.cuda()
            image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
            feats1, feats2 = torch.Tensor(feats1).cuda(), torch.Tensor(feats2).cuda()
            feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)
            labels = labels.unsqueeze(1).float()
            #print(image1.shape, image2.shape, feats1.shape, feats2.shape, labels.shape)

            ## TTA
            if len(image1.size())==5:
                batch_size, n_crops, c, h, w = image1.size()
                image1 = image1.view(-1, c, h, w)
                image2 = image2.view(-1, c, h, w)
                if FEATS:
                    _, n_feats = feats1.size()
                    feats1 = feats1.repeat(1,n_crops).view(-1,n_feats)
                    feats2 = feats2.repeat(1,n_crops).view(-1,n_feats)
                out = model(image1, feats1, image2, feats2)
                out = out.view(batch_size, n_crops, -1).mean(1)
            else:
                out = model(image1, feats1, image2, feats2)

            out = torch.sigmoid(out)
            res_prob += out.cpu().numpy().tolist()
            res_label += labels.cpu().numpy().tolist()
            res_name += [pid]

        res_label = np.array(res_label)
        res_prob = np.array(res_prob)
        test_auc = roc_auc_score(res_label, res_prob)
        test_acc = accuracy_score(res_label, (res_prob>0.5).astype(int))
        test_accs.append(test_acc)
        test_aucs.append(test_auc)
        print('Accuracy:', test_acc, 'AUC:', test_auc)
        y_pred += res_prob

# Test scores
y_pred /= FOLDS
y_true = np.array(res_label)
acc = accuracy_score(y_true, (y_pred>0.5).astype(int))
auc = roc_auc_score(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)
test_accs = np.array(test_accs)
test_aucs = np.array(test_aucs)
print('\nOverall Accuracy:', acc, 'AUC:', auc)
print('Accuracy mean:', np.mean(test_accs), 'std:', np.std(test_accs))
print('AUC mean:', np.mean(test_aucs), 'std:', np.std(test_aucs))
