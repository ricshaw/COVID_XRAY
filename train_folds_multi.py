import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2
import sys
import random
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage import color
from sklearn import linear_model
from sklearn import metrics
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
if False:
    import runai.hpo
    strategy = runai.hpo.Strategy.GridSearch
    runai.hpo.init('/nfs/project/richard', 'bloods-only')
    config = runai.hpo.pick(
    grid=dict(
            batch_size=[32,64,128],
            lr=[0.1,0.01,0.001],
            aug=[0.1,0.2,0.3],
            chns=[64,128,256],
            dropout=[0.1,0.2,0.3,0.4,0.5],
            images_size=[256,512],
            encoder=['efficientnet-b0']),
    strategy=strategy)
else:
    config = dict(
                batch_size=16,
                lr=0.001,
                aug=0.1,
                chns=128,
                dropout=0.1,
                image_size=512,
                encoder='efficientnet-b3',)
print(config)
data = '/nfs/home/richard/COVID_XRAY/cxr_news2_pseudonymised_filenames_folds.csv'
test_data = '/nfs/home/richard/COVID_XRAY/new_gstt.csv'

IMAGES = True
FEATS = False
MULTI = False
LATEST = False
VAL_MODE = 'ALL'

EPOCHS = 30
MIN_EPOCHS = 30
PATIENCE = 0
FOLDS = 5
GAMMA = 2.0
CUTMIX_PROB = 0.0
CENTRE_CROP = True

SAVE = True
SAVE_NAME = 'model-bs%d-lr%.03f' % (config['batch_size'], config['lr'])
if IMAGES:
    SAVE_NAME += '-' + config['encoder'] + '-images-sz%d' % config['image_size']
if FEATS:
    SAVE_NAME += '-feats-chns%d-dp%.1f-aug%.1f' % (config['chns'], config['dropout'], config['aug'])
if MULTI:
    SAVE_NAME += '-multi'
if LATEST:
    SAVE_NAME += '-latest'
SAVE_PATH = '/nfs/home/richard/COVID_XRAY/' + SAVE_NAME
print(SAVE_NAME)

log_name = './runs/' + SAVE_NAME
writer = SummaryWriter(log_dir=log_name)
if SAVE:
    os.makedirs(SAVE_PATH, exist_ok=True)


def get_latest(df):
    df['CXR_datetime'] = pd.to_datetime(df.CXR_datetime, dayfirst=True)
    df = df.groupby('patient_pseudo_id').apply(pd.DataFrame.sort_values, 'CXR_datetime', ascending=False).reset_index(drop=True)
    return df.drop_duplicates(subset='patient_pseudo_id', keep='first').reset_index(drop=True)

## Load train data
df = pd.read_csv(data)
df = df.sort_values(by=['patient_pseudo_id'], ascending=True).reset_index(drop=True)
if LATEST:
    df = get_latest(df)
print('Train data:', df.shape)
print(df.head(20))
print('Number of images:', df.shape[0])
print('Died:', df[df.Died == 1].shape[0])
print('Survived:', df[df.Died == 0].shape[0])

## Load test data
test_df = pd.read_csv(test_data)
test_df = test_df.dropna(subset=['Filename'])
test_df['Filename'] = '/nfs/home/pedro/COVID/GSTT_JPGs_All/' + test_df['Filename'].astype(str)
test_df = test_df[test_df.Time_Mismatch < 2]
test_df = test_df.reset_index(drop=True)
if LATEST:
    test_df = get_latest(test_df)
print('Test data:', test_df.shape)
print(test_df.head(20))

if FEATS:
    tmp = test_df.drop(columns=['patient_pseudo_id','CXR_datetime','Age','Gender','Ethnicity','Died','Time_Mismatch','Filename'])
    bloods_cols = tmp.columns
    #bloods_cols = ['Height', 'Weight', 'BMI', 'Albumin', 'Bilirubin', 'Creatinine', 'CRP']
    print('Bloods:', bloods_cols)

def new_section(text):
    print('\n')
    print(100 * '*')
    print(text)

def prepare_data(df, bloods_cols):
    print('Preparing data')
    ## Gender
    df['Male'] = 0
    df['Female'] = 0
    df.loc[df['Gender'] == 1, 'Male'] = 1
    df.loc[df['Gender'] == 0, 'Female'] = 1
    ## Age
    df.Age.replace(120, np.nan, inplace=True)
    ## Ethnicity
    df.Ethnicity.replace(np.nan, 'Unknown', inplace=True)
    #df.Ethnicity.replace('Unknown', np.nan, inplace=True)
    #df.Ethnicity.replace('White', 1, inplace=True)
    #df.Ethnicity.replace('Black', 2, inplace=True)
    #df.Ethnicity.replace('Asian', 3, inplace=True)
    #df.Ethnicity.replace('Mixed', 4, inplace=True)
    #df.Ethnicity.replace('Other', 5, inplace=True)
    df['White'] = 0
    df['Black'] = 0
    df['Asian'] = 0
    df['Mixed'] = 0
    df['Other'] = 0
    df.loc[df['Ethnicity'] == 'White', 'White'] = 1
    df.loc[df['Ethnicity'] == 'Black', 'Black'] = 1
    df.loc[df['Ethnicity'] == 'Asian', 'Asian'] = 1
    df.loc[df['Ethnicity'] == 'Mixed', 'Mixed'] = 1
    df.loc[df['Ethnicity'] == 'Other', 'Other'] = 1
    # Compute relative timestamps
    df_tmp = df.pop('CXR_datetime')
    df['CXR_datetime'] = df_tmp
    df['CXR_datetime'] = pd.to_datetime(df.CXR_datetime, dayfirst=True)
    df['min_datetime'] = df['CXR_datetime']
    df['min_datetime'] = df.groupby('patient_pseudo_id')['min_datetime'].transform('min')
    df['rel_datetime'] = (df['CXR_datetime'] - df['min_datetime']) / np.timedelta64(1,'D')

    # Extract features
    bloods = df.loc[:,bloods_cols].values.astype(np.float32)
    print('Bloods', bloods.shape)
    age = df.Age.values[:,None]
    #gender = df.Gender.values[:,None]
    #ethnicity = df.Ethnicity.values[:,None]
    time = df.rel_datetime.values[:,None]

    # Normalise features
    #X = np.concatenate((bloods, age, gender, ethnicity, time), axis=1)
    X = np.concatenate((bloods, age, time), axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Fill missing
    #print('Features before', np.nanmin(X), np.nanmax(X))
    #print('Missing before: %d' % sum(np.isnan(X).flatten()))
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(X)
    X = imputer.transform(X)
    #print('Features after', np.nanmin(X), np.nanmax(X))
    #print('Missing after: %d' % sum(np.isnan(X).flatten()))

    # Put back features
    df.loc[:,bloods_cols] = X[:,0:bloods.shape[1]]
    df.loc[:,'Age'] = X[:,bloods.shape[1]]
    #df.loc[:,'Gender'] = X[:,bloods.shape[1]+1]
    #df.loc[:,'Ethnicity'] = X[:,bloods.shape[1]+2]
    #df.loc[:,'rel_datetime'] = X[:,bloods.shape[1]+3]
    df.loc[:,'rel_datetime'] = X[:,bloods.shape[1]+1]
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
    #age = df.Age[i].astype(np.float32)
    #gender = df.Gender[i].astype(np.float32)
    #ethnicity = df.Ethnicity[i].astype(np.float32)
    male = df.Male[i].astype(np.float32)
    female = df.Female[i].astype(np.float32)
    age = df.Age[i].astype(np.float32)
    white = df.White[i].astype(np.float32)
    black = df.Black[i].astype(np.float32)
    asian = df.Asian[i].astype(np.float32)
    mixed = df.Mixed[i].astype(np.float32)
    other = df.Other[i].astype(np.float32)
    bloods = df.loc[i, bloods_cols].values.astype(np.float32)
    if aug:
        bloods += np.random.normal(0, config['aug'], bloods.shape)
    #feats = np.concatenate((bloods, [age, gender, ethnicity]), axis=0)
    feats = np.concatenate((bloods, [male, female, age, white, black, asian, mixed, other]), axis=0)
    return feats

def get_image(df, i, transform, A_transform=None):
    image = default_image_loader(df.Filename[i])
    # Centre crop
    if CENTRE_CROP:
        image = transforms.CenterCrop(min(image.size))(image)
    # A transform
    if A_transform is not None:
        image = np.array(image)
        image = A_transform(image=image)['image']
        image = Image.fromarray(image)
    # Transform
    image = transform(image)
    return image

class SingleImageDataset(Dataset):
    def __init__(self, my_df, transform, A_transform=None, unique=False):
        self.df = my_df
        self.unique_df = self.df.drop_duplicates(subset='patient_pseudo_id').reset_index(drop=True)
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform
        self.unique = unique

    def __getitem__(self, index):
        image, feats = np.array([]), np.array([])
        if self.unique:
            pid = self.unique_df.patient_pseudo_id[index]
            pid_df = self.df[self.df['patient_pseudo_id']==pid].reset_index(drop=True)
            pid_df = pid_df.loc[np.random.choice(pid_df.shape[0], 1, replace=True)].reset_index(drop=True)
            if IMAGES:
                image = get_image(pid_df, 0, self.transform, self.A_transform)
            if FEATS:
                feats = get_feats(pid_df, 0, aug=True)
                time = pid_df.rel_datetime[0].astype(np.float32)
                feats = np.append(feats, time)
            label = pid_df.Died[0]

        else:
            pid = self.df.patient_pseudo_id[index]
            if IMAGES:
                image = get_image(self.df, index, self.transform, self.A_transform)
            if FEATS:
                feats = get_feats(self.df, index, aug=True)
                time = self.df.rel_datetime[index].astype(np.float32)
                feats = np.append(feats, time)
            label = self.df.Died[index]
        return pid, image, feats, label

    def __len__(self):
        if self.unique:
            return self.unique_df.shape[0]
        else:
            return self.df.shape[0]


class MultiImageDataset(Dataset):
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
            image1 = get_image(pid_df, 0, self.transform, self.A_transform)
            image2 = get_image(pid_df, 1, self.transform, self.A_transform)
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
    def __init__(self, encoder='efficientnet-b0', nfeats=38):
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
            hidden1 = config['chns']
            hidden2 = config['chns']
            self.out_chns += hidden2
            self.fc1 = nn.Linear(nfeats, hidden1, bias=False)
            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.meta = nn.Sequential(self.fc1,
                                      nn.BatchNorm1d(hidden1),
                                      #nn.ReLU(),
                                      Swish_Module(),
                                      nn.Dropout(p=config['dropout']),
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
        hidden1 = config['chns']
        self.classifier = nn.Sequential( #nn.ReLU(),
                                         Swish_Module(),
                                         nn.Dropout(p=config['dropout']),
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
                         A.Resize(config['image_size'], config['image_size'], interpolation=3, p=1),
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=3, border_mode=4, p=0.5),
                         #A.Rotate(p=1, limit=45, interpolation=3),
                         #A.RandomResizedCrop(config['image_size'], config['image_size'], scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=3, p=1),
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
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
val_transform = transforms.Compose([
                              transforms.Resize(config['image_size'], 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
tta_transform = transforms.Compose([transforms.Resize(config['image_size'], 3),
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
    check_dataset = SingleImageDataset(df, train_transform, A_transform, unique=False)
    check_loader = DataLoader(check_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
    for i, sample in enumerate(check_loader):
        pid, image, feats, label = sample[0], sample[1], sample[2], sample[3]
        print('\n',pid[0])
        print('image:', image.shape, 'feats:', feats.shape, 'label:', label.shape)
    exit(0)

if False:
    check_dataset = MultiImageDataset(df, train_transform, A_transform)
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
        #cv2.imwrite(out_name, out_image)
    exit(0)

## Init
val_preds = []
val_labels = []
val_names = []
val_acc = []
val_auc = []
val_folds = []
scalers = []

## Fold loop
for fold in range(FOLDS):
    new_section('Fold: %d' % fold)

    ## Init datasets
    train_df = df[df.fold != fold].reset_index(drop=True, inplace=False)
    val_df = df[df.fold == fold].reset_index(drop=True, inplace=False)

    ## Prepare data
    if FEATS:
        train_df = prepare_data(train_df, bloods_cols).reset_index(drop=True, inplace=False)
        val_df = prepare_data(val_df, bloods_cols).reset_index(drop=True, inplace=False)
    print('Train', train_df.shape)
    print('Valid', val_df.shape)

    ## Train dataloader
    if MULTI:
        train_dataset = MultiImageDataset(train_df, train_transform, A_transform)
    else:
        train_dataset = SingleImageDataset(train_df, train_transform, A_transform, unique=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True, drop_last=True)

    ## Val dataloader
    val_dataset = SingleImageDataset(val_df, tta_transform, A_transform=None, unique=False)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)

    ## Init model
    singlemodel = SingleModel(config['encoder']).cuda()
    model = CombinedModel(singlemodel).cuda()
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters(), lr=config['lr'])
    ALPHA = train_df[train_df.Died==0].shape[0]/train_df.shape[0]
    print('Alpha', ALPHA)

    ## Init metrics
    running_acc = []
    running_auc = []
    running_preds = []
    best_auc = 0.0
    stop_count = 0

    ## Training loop
    for epoch in range(EPOCHS):

        # Training step
        print('\nTraining step')
        model.cuda()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        res_name, res_prob, res_label = [], [], []
        for i, sample in enumerate(train_loader):
            pid = sample[0]
            if MULTI:
                image1, image2 = sample[1].cuda(), sample[2].cuda()
                feats1, feats2 = sample[3].cuda(), sample[4].cuda()
                time1, time2 = sample[5], sample[6]
                labels = sample[7].cuda()
            else:
                image1, image2 = sample[1].cuda(), None
                feats1, feats2 = sample[2].cuda(), None
                labels = sample[3].cuda()
            labels = labels.unsqueeze(1).float()
            #print('image1', image1, 'feats1', feats1, 'image2', image2, 'feats2', feats2)

            ## CUTMIX
            prob = np.random.rand(1)
            if prob < CUTMIX_PROB:
                # generate mixed sample
                lam = np.random.beta(1,1)
                rand_index = torch.randperm(image1.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                features_a = features
                features_b = features[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(image1.size(), lam)
                image1[:, :, bbx1:bbx2, bby1:bby2] = image1[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image1.size()[-1] * image1.size()[-2]))
                features = features_a * lam + features_b * (1. - lam)
                # compute output
                out = model(image1, feats1, image2, feats2)
                #loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
                loss = sigmoid_focal_loss(out, target_a, ALPHA, GAMMA, reduction="mean") * lam + \
                       sigmoid_focal_loss(out, target_b, ALPHA, GAMMA, reduction="mean") * (1. - lam)

            else:
                out = model(image1, feats1, image2, feats2)
                #loss = criterion(out, labels)
                loss = sigmoid_focal_loss(out, labels, alpha=ALPHA, gamma=GAMMA, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.5).int() == labels).sum().item()
            res_prob += out.detach().cpu().numpy().tolist()
            res_label += labels.detach().cpu().numpy().tolist()

        # Scores
        y_true = np.array(res_label)
        y_pred = np.array(res_prob)
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        print("Epoch: {}, Loss: {}, Train Accuracy: {}, AUC: {}".format(epoch, running_loss, round(acc, 4), round(auc, 4)))

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

        ## Validation step
        print('\nValidation step')
        model.eval()
        running_loss = 0
        correct, total = 0, 0
        res_name, res_prob, res_label = [], [], []
        with torch.no_grad():
            if MULTI:
                for pid in val_df['patient_pseudo_id'].unique():
                    pid_df = val_df[val_df['patient_pseudo_id']==pid].reset_index(drop=True)
                    pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                    pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                    n_images = pid_df.shape[0]
                    if VAL_MODE == 'ALL':
                        inds = range(n_images)
                    elif VAL_MODE == 'FIRST_LAST':
                        inds = [0, n_images-1]
                    elif VAL_MODE == 'LATEST':
                        inds = [n_images-1, n_images-1]
                    n = len(inds)
                    for i in range(n):
                        ind1 = inds[0] # Always first image
                        ind2 = inds[i] # Next image
                        time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                        time2 = pid_df.rel_datetime[ind2].astype(np.float32)
                        image1, image2 = np.array([]), np.array([])
                        feats1, feats2 = np.array([]), np.array([])
                        if FEATS:
                            feats1 = get_feats(pid_df, ind1, aug=False)
                            feats2 = get_feats(pid_df, ind2, aug=False)
                            feats1 = np.append(feats1, time1)
                            feats2 = np.append(feats2, time2)
                        else:
                            feats1, feats2 = np.array([]), np.array([])
                        if IMAGES:
                            image1 = get_image(pid_df, ind1, tta_transform)
                            image2 = get_image(pid_df, ind2, tta_transform)
                        else:
                            image1, image2 = torch.FloatTensor(), torch.FloatTensor()
                        labels = pid_df.Died[ind1]
                        labels = torch.Tensor([labels]).cuda()
                        image1, image2 = image1.cuda(), image2.cuda()
                        image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
                        feats1, feats2 = torch.Tensor(feats1).cuda(), torch.Tensor(feats2).cuda()
                        feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)
                        labels = labels.unsqueeze(1).float()
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
                        loss = sigmoid_focal_loss(out, labels, alpha=ALPHA, gamma=GAMMA, reduction="mean")
                        running_loss += loss.item()
                        out = torch.sigmoid(out).item()
                        correct += int(int(out>0.5)==labels.item())
                        total += 1
                        res_prob += [out]
                        res_label += [labels.item()]
                        res_name += [pid]
            else:
                for i, sample in enumerate(val_loader):
                    pid = sample[0]
                    image1, feats1 = sample[1].cuda(), sample[2].cuda()
                    labels = sample[3].cuda()
                    labels = labels.unsqueeze(1).float()
                    ## TTA
                    if len(image1.size())==5:
                        batch_size, n_crops, c, h, w = image1.size()
                        image1 = image1.view(-1, c, h, w)
                        if FEATS:
                            _, n_feats = feats1.size()
                            feats1 = feats1.repeat(1,n_crops).view(-1,n_feats)
                        out = model(image1, feats1, image2=None, feats2=None)
                        out = out.view(batch_size, n_crops, -1).mean(1)
                    else:
                        out = model(image1, feats1, image2=None, feats2=None)
                    loss = sigmoid_focal_loss(out, labels, alpha=ALPHA, gamma=GAMMA, reduction="mean")
                    running_loss += loss.item()
                    total += labels.size(0)
                    out = torch.sigmoid(out)
                    correct += ((out>0.5).int() == labels).sum().item()
                    res_prob += out.cpu().numpy().tolist()
                    res_label += labels.cpu().numpy().tolist()
                    res_name += pid
                res_prob = [x[0] for x in res_prob]
                res_label = [x[0] for x in res_label]

        # Scores
        y_true = np.array(res_label)
        y_pred = np.array(res_prob)
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        running_auc.append(auc)
        running_acc.append(acc)
        running_preds.append(y_pred)
        id = int(np.argmax(running_auc))
        best_epoch = id
        print("Epoch: {}, Loss: {}, Test Accuracy: {}, AUC: {}".format(epoch, running_loss, round(acc,4), round(auc,4)))
        print('All AUC:', running_auc)
        print('Best Result -- Epoch:', id, 'AUC:', round(running_auc[id],4), 'Accuracy:', round(running_acc[id],4))

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
            val_acc += [running_acc[id]]
            val_auc += [running_auc[id]]
            val_labels += res_label
            val_names += res_name
            val_folds += len(res_name)*[fold]
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, epoch)))
                torch.save(model.state_dict(), MODEL_PATH)
            del model
            break

## Totals
val_labels = np.array(val_labels)
val_preds = np.array(val_preds)
val_auc = np.array(val_auc)
val_acc = np.array(val_acc)
print('Labels:', len(val_labels), 'Preds:', len(val_preds), 'Names:', len(val_names), 'AUCs:', len(val_auc))
acc = balanced_accuracy_score(val_labels, (val_preds>0.5).astype(int))
auc = roc_auc_score(val_labels, val_preds)
print("Total Accuracy: {}, AUC: {}".format(round(acc,4), round(auc,4)))
print('Accuracy mean:', np.mean(val_acc), 'std:', np.std(val_acc))
print('AUC mean:', np.mean(val_auc), 'std:', np.std(val_auc))

## ROC curve
fpr, tpr, _ = roc_curve(val_labels, val_preds)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Prediction of Death - Precision-Recall')
plt.legend(loc="lower right")
plt.savefig(os.path.join(SAVE_PATH,'precision-recall-' + SAVE_NAME + '.png'), dpi=300)

## Preds dataframe
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels, "Pred":val_preds, "Fold":val_folds})
sub.to_csv(os.path.join(SAVE_PATH,'preds-KCH-' + SAVE_NAME + '.csv'), index=False)
exit(0)


## Test
new_section('Testing!')
print(SAVE_NAME)
test_df = prepare_data(test_df, bloods_cols)
print('Test data:', test_df.shape)

if not MULTI:
    test_dataset = SingleImageDataset(test_df, tta_transform, unique=False)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

y_pred = 0
test_accs = []
test_aucs = []
sub = pd.DataFrame(columns=['Filename','Died','Pred'])
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
        if MULTI:
            for pid in test_df['patient_pseudo_id'].unique():
                pid_df = test_df[test_df['patient_pseudo_id']==pid].reset_index(drop=True)
                pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                n_images = pid_df.shape[0]
                if VAL_MODE == 'ALL':
                    inds = range(n_images)
                elif VAL_MODE == 'FIRST_LAST':
                    inds = [0, n_images-1]
                elif VAL_MODE == 'LATEST':
                    inds = [n_images-1, n_images-1]
                n = len(inds)
                for i in range(n):
                    ind1 = inds[0] # Always first image
                    ind2 = inds[i] # Next image
                    time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                    time2 = pid_df.rel_datetime[ind2].astype(np.float32)
                    image1, image2 = np.array([]), np.array([])
                    feats1, feats2 = np.array([]), np.array([])
                    if FEATS:
                        feats1 = get_feats(pid_df, ind1, aug=False)
                        feats2 = get_feats(pid_df, ind2, aug=False)
                        feats1 = np.append(feats1, time1)
                        feats2 = np.append(feats2, time2)
                    else:
                        feats1, feats2 = np.array([]), np.array([])
                    if IMAGES:
                        image1 = get_image(pid_df, ind1, tta_transform)
                        image2 = get_image(pid_df, ind2, tta_transform)
                    else:
                        image1, image2 = torch.FloatTensor(), torch.FloatTensor()
                    labels = pid_df.Died[ind1]
                    image1, image2 = image1.cuda(), image2.cuda()
                    image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
                    feats1, feats2 = torch.Tensor(feats1).cuda(), torch.Tensor(feats2).cuda()
                    feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)
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
                    out = torch.sigmoid(out).item()
                    res_prob += [out]
                    res_label += [labels]
                    res_name += [pid]
        else:
            for i, sample in enumerate(test_loader):
                pid, image, feats, labels = sample[0], sample[1].cuda(), sample[2].cuda(), sample[3].cuda()
                labels = labels.unsqueeze(1).float()
                ## TTA
                if len(image1.size())==5:
                    batch_size, n_crops, c, h, w = image1.size()
                    image1 = image1.view(-1, c, h, w)
                    if FEATS:
                        _, n_feats = feats1.size()
                        feats1 = feats1.repeat(1,n_crops).view(-1,n_feats)
                    out = model(image, feats, image2=None, feats2=None)
                    out = out.view(batch_size, n_crops, -1).mean(1)
                else:
                    out = model(image, feats, image2=None, feats2=None)
                out = torch.sigmoid(out)
                res_prob += out.cpu().numpy().tolist()
                res_label += labels.cpu().numpy().tolist()
                res_name += pid.cpu().numpy().tolist()
            res_prob = [x[0] for x in res_prob]
            res_label = [x[0] for x in res_label]

        res_label = np.array(res_label)
        res_prob = np.array(res_prob)
        test_auc = roc_auc_score(res_label, res_prob)
        test_acc = balanced_accuracy_score(res_label, (res_prob>0.5).astype(int))
        test_accs.append(test_acc)
        test_aucs.append(test_auc)
        print('Accuracy:', test_acc, 'AUC:', test_auc)
        y_pred += res_prob
        sub['Fold %d' % fold] = res_prob

# Test scores
y_pred /= FOLDS
y_true = np.array(res_label)
acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
auc = roc_auc_score(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)
test_accs = np.array(test_accs)
test_aucs = np.array(test_aucs)
print('\nOverall Accuracy:', acc, 'AUC:', auc)
print('Accuracy mean:', np.mean(test_accs), 'std:', np.std(test_accs))
print('AUC mean:', np.mean(test_aucs), 'std:', np.std(test_aucs))
sub['Filename'] = res_name
sub['Died'] = y_true
sub['Pred'] = y_pred
sub.to_csv(os.path.join(SAVE_PATH,'preds-GSTT-' + SAVE_NAME + '.csv'), index=False)

## Report
val_acc_mean = np.asscalar(np.mean(val_acc))
val_acc_std = np.asscalar(np.std(val_acc))
val_auc_mean = np.asscalar(np.mean(val_auc))
val_auc_std = np.asscalar(np.std(val_auc))

test_acc_mean = np.asscalar(np.mean(test_accs))
test_acc_std = np.asscalar(np.std(test_accs))
test_auc_mean = np.asscalar(np.mean(test_aucs))
test_auc_std = np.asscalar(np.std(test_aucs))
runai.hpo.report(epoch=best_epoch, metrics={ 'val_acc':val_acc_mean, 'val_acc_std':val_acc_std,
                                             'val_auc':val_auc_mean, 'val_auc_std':val_auc_std,
                                             'test_acc':test_acc_mean, 'test_acc_std':test_acc_std,
                                             'test_auc':test_auc_mean, 'test_auc_std':test_auc_std })
