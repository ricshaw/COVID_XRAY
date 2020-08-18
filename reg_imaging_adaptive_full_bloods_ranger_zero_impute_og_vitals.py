import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage.transform import resize
import datetime
import glob

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import argparse
import albumentations as A
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from robust_loss_pytorch import AdaptiveLossFunction
import sys
sys.path.append('/nfs/home/pedro/RangerLARS/over9000')
from over9000 import RangerLars

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


parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--labels', nargs='+', type=str)
parser.add_argument('--images_dir', nargs='+', type=str)
parser.add_argument('--job_name', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--resolution', type=int)
arguments = parser.parse_args()


# Writer will output to ./runs/ directory by default
log_dir = f'/nfs/home/pedro/COVID/logs/{arguments.job_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# writer = SummaryWriter(log_dir=log_dir)


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img


def image_normaliser(some_image):
    return 1 * (some_image - torch.min(some_image)) / (torch.max(some_image) - torch.min(some_image))


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
        # Augmentations
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)
        image = self.transform(image)
        image = np.array(image)

        label = self.df['Inv_Time_To_Death'][index]

        first_blood = '.cLac'
        last_blood = 'OBS BMI Calculation'
        bloods = self.df.loc[index, first_blood:last_blood].values.astype(np.float32)
        first_vital = 'Fever (finding)'
        last_vital = 'Immunodeficiency disorder (disorder)'
        vitals = self.df.loc[index, first_vital:last_vital].values.astype(np.float32)
        age = self.df.Age[index][..., None]
        gender = self.df.Gender[index][..., None]
        ethnicity = self.df.Ethnicity[index][..., None]
        days_from_onset_to_scan = self.df['days_from_onset_to_scan'][index][..., None]

        features = np.concatenate((bloods, age, gender, ethnicity, days_from_onset_to_scan, vitals), axis=0)
        return image, filepath, label, features

    def __len__(self):
        return self.df.shape[0]


def factor_int(n):
    nsqrt = np.ceil(np.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val -= 1
    return int(val), int(val2)


# Some necessary variables
img_dir = arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
labels = arguments.labels  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(labels)
SAVE_PATH = os.path.join(f'/nfs/home/pedro/COVID/models/{arguments.job_name}')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
SAVE = True
LOAD = True

# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)

# Hyperparameter loading: General parameters so doesn't matter which model file is loaded exactly
if LOAD and num_files > 0:
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    latest_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
    print(f'Loading {latest_model_file}')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    # Extras that may not exist in older models
    bs = checkpoint['batch_size']
    input_size = checkpoint['resolution']
    EPOCHS = 1000
    FOLDS = 5
else:
    running_iter = 0
    loaded_epoch = -1
    if arguments.resolution < 500:
        bs = 32
    else:
        bs = 16
    input_size = (arguments.resolution, arguments.resolution)  # (528, 528)
    EPOCHS = 100
    FOLDS = 5

# Load labels
print(f'The  labels are {labels}')
if len(labels) == 1:
    labels = labels[0]
    df = pd.read_csv(labels)

    death_dates = df['date of death']
    filenames = df['Filename']
    time_differences = []
    for ID, filename in enumerate(filenames):
        scan_time = filename.split('_')[1]
        scan_date = datetime.datetime(year=int(scan_time[0:4]),
                                      month=int(scan_time[4:6]),
                                      day=int(scan_time[6:8]))
        # print(death_dates[ID])
        if death_dates[ID] == '00/01/1900' or pd.isnull(death_dates[ID]):
            time_string = '01/01/1900'
        else:
            time_string = death_dates[ID]
        death_date = datetime.datetime.strptime(time_string, "%d/%m/%Y")
        time_difference = abs((death_date - scan_date).days) + 1
        time_differences.append(time_difference)
    df['Time_To_Death'] = time_differences
    df['Inv_Time_To_Death'] = 1/df['Time_To_Death']
    df.loc[(df['Inv_Time_To_Death'] < 1e-3), 'Inv_Time_To_Death'] = 0.0
    df['Filename'] = img_dir[0] + '/' + df['Filename'].astype(str)


elif len(labels) > 1:
    df = pd.read_csv(labels[0])
    for extra in range(1, len(labels)):
        extra_df = pd.read_csv(labels[extra])
        df = pd.concat([df, extra_df], ignore_index=True)

## Replace data
df.Age.replace(120, np.nan, inplace=True)
df.Ethnicity.replace('Unknown', np.nan, inplace=True)
df.Ethnicity.replace('White', 1, inplace=True)
df.Ethnicity.replace('Black', 2, inplace=True)
df.Ethnicity.replace('Asian', 3, inplace=True)
df.Ethnicity.replace('Mixed', 4, inplace=True)
df.Ethnicity.replace('Other', 5, inplace=True)

# Onset date days
df['days_from_onset_to_scan'] = df['days_from_onset_to_scan'].astype(float)
df['days_from_onset_to_scan'] = df['days_from_onset_to_scan'].apply(lambda x: x if x <= 1000 else np.nan)

# Correct gas bloods: .pO2, .pCO2, cHCO3, .pH
df.loc[df['.pO2'] < 6, '.pCO2'] = np.nan
df.loc[df['.pO2'] < 6, '.cHCO3'] = np.nan
df.loc[df['.pO2'] < 6, '.pH'] = np.nan
df.loc[df['.pO2'] < 6, '.pO2'] = np.nan

# Extract features
first_blood = '.cLac'
last_blood = 'OBS BMI Calculation'
bloods = df.loc[:, first_blood:last_blood].values.astype(np.float32)
print('Bloods', bloods.shape)
first_vital = 'Fever (finding)'
last_vital = 'Immunodeficiency disorder (disorder)'
vitals = df.loc[:, first_vital:last_vital].values.astype(np.float32)

print('Vitals', vitals.shape)
age = df.Age[:, None]
gender = df.Gender[:, None]
ethnicity = df.Ethnicity[:, None]
days_from_onset_to_scan = df['days_from_onset_to_scan'][:, None]

# Normalise features
scaler = StandardScaler()
X = np.concatenate((bloods, age, gender, ethnicity, days_from_onset_to_scan), axis=1)
scaler.fit(X)
X = scaler.transform(X)
X = np.concatenate((X, vitals), axis=1)

# Fill missing
print('Features before', np.nanmin(X), np.nanmax(X))
print('Missing before: %d' % sum(np.isnan(X).flatten()))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value=0)
imputer.fit(X)
X = imputer.transform(X)
print('Features after', np.nanmin(X), np.nanmax(X))
print('Missing after: %d' % sum(np.isnan(X).flatten()))
df.loc[:, first_blood:last_blood] = X[:, 0:bloods.shape[1]]
df.loc[:, first_vital:last_vital] = X[:, bloods.shape[1]:bloods.shape[1] + vitals.shape[1]]
df.loc[:, 'Age'] = X[:, -4]
df.loc[:, 'Gender'] = X[:, -3]
df.loc[:, 'Ethnicity'] = X[:, -2]
df.loc[:, 'days_from_onset_to_scan'] = X[:, -1]

df.loc[:, first_blood:last_blood] = X[:, 0:bloods.shape[1]]
df.loc[:, first_vital:last_vital] = X[:, bloods.shape[1]+4:bloods.shape[1]+4 + vitals.shape[1]]
df.loc[:, 'Age'] = X[:, bloods.shape[1]]
df.loc[:, 'Gender'] = X[:, bloods.shape[1]+1]
df.loc[:, 'Ethnicity'] = X[:, bloods.shape[1]+2]
df.loc[:, 'days_from_onset_to_scan'] = X[:, bloods.shape[1]+3]

# For shape purposes:
first_blood = '.cLac'
last_blood = 'OBS BMI Calculation'
bloods = df.loc[:, first_blood:last_blood]
first_vital = 'Fever (finding)'
last_vital = 'Immunodeficiency disorder (disorder)'
vitals = df.loc[:, first_vital:last_vital]
age = df.Age
gender = df.Gender
ethnicity = df.Ethnicity
days_from_onset_to_scan = df['days_from_onset_to_scan']
temp_bloods = pd.concat([bloods, age, gender, ethnicity, days_from_onset_to_scan, vitals], axis=1, sort=False)


# Pre-processing transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(input_size, 3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

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
                                        transforms.ToTensor()(
                                            image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(90, resample=0)),
                                        transforms.ToTensor()(
                                            image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(180, resample=0)),
                                        transforms.ToTensor()(
                                            image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(270, resample=0)),
                                    ])),
                                    transforms.Lambda(lambda images: torch.stack(
                                        [transforms.Normalize(mean, std)(image) for image in images]))
                                    ])
# Augmentations
A_transform = A.Compose([
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.Rotate(p=1, limit=45, interpolation=3),
                         A.OneOf([
                                  A.IAAAdditiveGaussianNoise(),
                                  A.GaussNoise(),
                                 ], p=0.25),
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
                                 ], p=0.1),
                         A.OneOf([
                                  A.CLAHE(clip_limit=2),
                                  A.IAASharpen(),
                                  A.IAAEmboss(),
                                  A.RandomBrightnessContrast(),
                                 ], p=1),
                         A.RandomGamma(p=1),
                        ], p=1)


class Model(nn.Module):
    def __init__(self, encoder='efficientnet-b3'):
        super(Model, self).__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.n_channels_dict = n_channels_dict
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
        # self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = EfficientNet.from_pretrained(encoder, num_classes=1)
        n_feats = len(temp_bloods.columns)
        hidden1 = 256
        hidden2 = 256
        dropout = 0.3
        self.fc1 = nn.Linear(n_feats, hidden1, bias=True)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.meta = nn.Sequential(self.fc1,
                                  # nn.BatchNorm1d(hidden1),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  self.fc2,
                                  # nn.BatchNorm1d(hidden2),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout)
                                  )

        self.classifier = nn.Sequential(nn.Linear(self.n_channels_dict[encoder] + hidden2, out_features=1, bias=True),
                                        nn.Softplus())

    def forward(self, x, features):
        # Images
        x = self.net.extract_features(x)

        # Pooling and final linear layer
        x = self.avg_pool(x)
        x = nn.Flatten()(x)

        y = self.meta(features)

        z = torch.cat((x, y), dim=1)
        out = self.classifier(z)

        # out = self.net(x)
        return out

use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')

# For aggregation
overall_val_preds = []
overall_val_labels = []
overall_val_names = []

overall_val_mse = []
overall_val_mae = []
overall_mvp_features = []


# If pretrained then initial model file will NOT match those created here: Therefore need to account for this
# Because won't be able to extract epoch and/ or fold from the name
if LOAD and num_files > 0:
    pretrained_checker = 'fold' in os.path.basename(latest_model_file)

# Find out fold and epoch
if LOAD and num_files > 0 and pretrained_checker:
    latest_epoch = int(os.path.splitext(os.path.basename(latest_model_file))[0].split('_')[2])
    latest_fold = int(os.path.splitext(os.path.basename(latest_model_file))[0].split('_')[4])
else:
    latest_epoch = -1
    latest_fold = 0


for fold in range(latest_fold, FOLDS):
    print('\nFOLD', fold)
    # Loss function
    adaptive_loss = True
    if adaptive_loss:
        criterion = AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device='cuda:0')
    else:
        criterion = torch.nn.MSELoss()

    # Running lists
    running_val_preds = []
    running_val_labels = []
    running_val_names = []
    running_val_mse = []
    running_val_mae = []
    running_mvp_features = []
    # Pre-loading sequence
    model = Model()
    # alpha = torch.FloatTensor([0.9, 0.8, 0.7, 0.25])[None, ...].cuda()
    optimizer = RangerLars(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    # Specific fold writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold}'))

    # Pretrained loading workaround
    if arguments.mode == 'pretrained':
        model.net._fc = nn.Linear(in_features=1536, out_features=14, bias=True)
        print(model.net._fc)

    # Load fold specific model
    if LOAD and num_files > 0 and arguments.mode == 'pretrained':
        # Get model file specific to fold
        # loaded_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
        # checkpoint = torch.load(os.path.join(SAVE_PATH, loaded_model_file), map_location=torch.device('cuda:0'))
        checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
        # Adjust key names
        keys_list = checkpoint['model_state_dict'].keys()
        new_dict = checkpoint['model_state_dict'].copy()
        for name in keys_list:
            new_dict[name[7:]] = checkpoint['model_state_dict'][name]
            del new_dict[name]
        model.load_state_dict(new_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # For pretrained networks, don't want first iteration to be loaded from pretraining
        running_iter = 0
        # Ensure that no more loading is done for future folds
        LOAD = False

    elif LOAD and num_files > 0 and arguments.mode != 'pretraining':
        # Get model file specific to fold
        loaded_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
        checkpoint = torch.load(os.path.join(SAVE_PATH, loaded_model_file), map_location=torch.device('cuda:0'))
        # Main model variables
        # Adjust key names
        keys_list = checkpoint['model_state_dict'].keys()
        new_dict = checkpoint['model_state_dict'].copy()
        for name in keys_list:
            new_dict[name[7:]] = checkpoint['model_state_dict'][name]
            del new_dict[name]
        model.load_state_dict(new_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Get the validation entries from previous folds!
        running_val_preds = checkpoint['running_val_preds']
        running_val_labels = checkpoint['running_val_labels']
        running_val_names = checkpoint['running_val_names']
        overall_val_preds = checkpoint['overall_val_preds']
        overall_val_labels = checkpoint['overall_val_labels']
        overall_val_names = checkpoint['overall_val_names']
        running_mvp_features = checkpoint['running_mvp_features']
        overall_mvp_features = checkpoint['overall_mvp_features']
        # Ensure that no more loading is done for future folds
        LOAD = False

    # Something extra to fix optimiser issues
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


    # # Pretrained loading workaround
    if arguments.mode == 'pretrained':
        model.net._fc = nn.Linear(in_features=1536, out_features=4, bias=True)
        print(model.net._fc)

        # Freeze parameters
        frozen_dudes = list(range(13))
        frozen_dudes = [str(froze) for froze in frozen_dudes]
        param_counter = 0
        for name, param in model.named_parameters():
            if any(f'.{frozen_dude}.' in name for frozen_dude in frozen_dudes) or param_counter < 3:
                param.requires_grad = False
            param_counter += 1

    model = nn.DataParallel(model)

    # Train / Val split
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print(f'The length of the training is {len(train_df)}')
    print(f'The length of the validation is {len(val_df)}')

    train_dataset = ImageDataset(train_df, transform, A_transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)

    val_dataset = ImageDataset(val_df, tta_transform)
    val_loader = DataLoader(val_dataset, batch_size=int(bs/4), num_workers=8)

    print(f'The shape of the labels are: {df.shape}')
    # for colu in df.columns:
    #     print(colu)
    # Best model selection
    best_val_mse = 1000
    best_counter = 0

    # Training
    if arguments.mode == 'train' or arguments.mode == 'pretrained':
        model.cuda()
        print('\nStarting training!')
        for epoch in range(latest_epoch+1, EPOCHS):
            print('Training step')
            running_loss = 0.0
            model.train()
            train_acc = 0
            total = 0

            for i, sample in enumerate(train_loader):
                images, names, labels, bloods = sample[0], sample[1], sample[2], sample[3]

                images = images.cuda()
                labels = labels.cuda()
                labels = labels.unsqueeze(1).float()
                bloods = bloods.cuda()
                bloods = bloods.float()

                out = model(images, bloods)
                # loss = criterion(out, labels)
                if adaptive_loss:
                    loss = torch.mean(criterion.lossfun(out-labels))
                else:
                    loss = criterion(out, labels)
                print(f'Labels are {labels[:10]}\n Preds are {out[:10]}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                total += labels.numel()

                train_acc += ((out - labels)**2).sum().item()
                # correct += ((out > 0.5).int() == labels).sum().item()

                # Name check: Shuffling sanity check
                if i == 0:
                    print(f'The test names are: {names[0]}, {names[-2]}')

                # Writing to tensorboard
                if running_iter % 50 == 0:
                    writer.add_scalar('Loss/train', loss.item(), running_iter)

                    # Reshaping grids: Same shape for both
                    labels_shape = factor_int(labels.shape[0])
                    # Convert labels and output to grid
                    labels_grid = torchvision.utils.make_grid(torch.reshape(labels, labels_shape))
                    output_grid = torchvision.utils.make_grid(torch.reshape(out, labels_shape))
                    images_grid = torchvision.utils.make_grid(images)
                    writer.add_image('Visuals/Images', image_normaliser(images_grid), running_iter)
                    writer.add_image('Visuals/Labels', image_normaliser(labels_grid), running_iter)
                    writer.add_image('Visuals/Output', image_normaliser(output_grid), running_iter)

                print("iter: {}, Loss: {}".format(running_iter, loss.item()))
                running_iter += 1

            print("Epoch: {}, Loss: {},\n Train Accuracy: {}".format(epoch, running_loss, train_acc/total))
            # if epoch % 2 == 1:
            #     scheduler.step()

            print('Validation step')
            model.eval()
            running_loss = 0
            # correct = 0
            val_counter = 0
            total = 0
            val_preds = []
            val_labels = []
            val_names = []

            with torch.no_grad():
                for images, names, labels, bloods in val_loader:
                    images = images.cuda()
                    labels = labels.cuda()
                    labels = labels.unsqueeze(1).float()
                    bloods = bloods.cuda()
                    bloods = bloods.float()

                    ## TTA
                    # print(bloods.size(), images.size())
                    batch_size, n_crops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)

                    # blud_batch_size, n_crops, h, w = bloods.size()
                    # bloods = bloods.view(-1, w)
                    # print(bloods.size(), images.size())
                    _, n_feats = bloods.size()
                    bloods = bloods.repeat(1, n_crops).view(-1, n_feats)
                    out = model(images, bloods)
                    # print(out.shape)
                    # print(out.view(batch_size, n_crops, -1).shape)
                    out = out.view(batch_size, n_crops, -1).mean(1)

                    if adaptive_loss:
                        val_loss = torch.mean(criterion.lossfun(out-labels))
                    else:
                        val_loss = criterion(out, labels)
                    running_loss += val_loss.item()

                    total += labels.numel()

                    # Save validation output for post all folds training aggregation
                    val_preds += out.cpu().numpy().tolist()
                    val_labels += labels.cpu().numpy().tolist()
                    val_names += names

                    acc = ((out - labels)**2).sum().item()
                    # correct += ((out > 0.5).int() == labels).sum().item()
                    val_counter += 1

            # Write to tensorboard
            writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)

            # acc = correct / total
            acc = ((out - labels)**2).sum().item()
            val_acc = acc / total
            y_true = np.array(val_labels)
            y_scores = np.array(val_preds)

            # Overalls
            def calc_mse(labels, scores):
                if type(labels) == np.ndarray:
                    return np.mean((scores - labels) ** 2)
                else:
                    return torch.mean((scores - labels)**2).item()

            def calc_mae(labels, scores):
                if type(labels) == np.ndarray:
                    return np.mean(np.abs(scores - labels))
                else:
                    return torch.mean(torch.abs(scores - labels)).item()

            # MSE and MAE calculations
            true_mse = calc_mse(y_true, y_scores)
            true_mae = calc_mse(y_true, y_scores)

            # Aggregation
            running_val_names.append(val_names)
            running_val_preds.append(val_preds)
            running_val_labels.append(val_labels)
            running_val_mse.append(true_mse)
            running_val_mae.append(true_mae)
            print("Epoch: {}, Loss: {},\n Test Accuracy: {},\n MSEs: {},\n MAEs {}\n".format(epoch,
                                                                                                    running_loss,
                                                                                                    val_acc,
                                                                                                    true_mse,
                                                                                                    true_mae))
            writer.add_scalar('Loss/MSE', true_mse, running_iter)
            writer.add_scalar('Loss/MAE', true_mae, running_iter)

            # Check if better than current best:
            if true_mse < best_val_mse:
                best_val_mse = true_mse
                append_string = 'best'
                best_counter = 0
            else:
                append_string = 'nb'
                best_counter += 1

            if append_string == 'best' and epoch > 5:
            # if (epoch == (EPOCHS - 1)) or (epoch % 10 == 0):
                occlusion = True
            else:
                occlusion = False
            if occlusion:
                print(f'Running occlusion on fold {fold}!')
                mvp_features = []
                occlusion_count = 0
                for images, names, labels, bloods in val_loader:
                    # Pick one
                    # random_index = np.random.randint(labels.size(0))
                    # names = names[random_index]
                    # name = os.path.basename(names)
                    # name = os.path.splitext(name)[0]
                    # labels = labels[random_index, ...][None, ...].cuda()
                    # print(label.shape, label)
                    # print(image.shape, image)
                    labels = labels.cuda()
                    labels = labels.unsqueeze(1).float()
                    # bloods = bloods[random_index, ...][None, ...]
                    bloods = bloods.cuda().float()
                    images = images[:, 0, ...].cuda()

                    # Account for tta: Take first image (non-augmented)
                    # Label does not need to be touched because it is obv. the same for this image regardless of tta
                    # Set a baseline
                    baseline = torch.zeros_like(images).cuda()
                    baseline_bloods = torch.zeros_like(bloods).cuda().float()

                    # Calculate attribution scores + delta
                    # ig = IntegratedGradients(model)
                    oc = Occlusion(model)
                    # nt = NoiseTunnel(ig)
                    # attributions, delta = nt.attribute(image, nt_type='smoothgrad', stdevs=0.02, n_samples=2,
                    #                                    baselines=baseline, target=0, return_convergence_delta=True)
                    _, target_ID = torch.max(labels, 1)
                    print(target_ID)
                    if occlusion_count == 0:
                        x_shape = 16
                        x_stride = 8
                    else:
                        x_shape = input_size[0]
                        x_stride = input_size[0]

                    oc_attributions0, blud0 = oc.attribute((images, bloods), sliding_window_shapes=((3, x_shape, x_shape), (1,)),
                                                    strides=((3, x_stride, x_stride), (1,)), target=0,
                                                    baselines=(baseline, baseline_bloods))                    # print('IG + SmoothGrad Attributions:', attributions)
                    # print('Convergence Delta:', delta)
                    image_grid = torchvision.utils.make_grid(images)

                    # Print
                    for single_feature in range(blud0.shape[0]):
                        mvp_feature = temp_bloods.columns[int(np.argmax(blud0[single_feature, :].cpu()))]
                        print(f'The most valuable feature was {mvp_feature}')
                        mvp_features.append(mvp_feature)

                    random_index = np.random.randint(labels.size(0))
                    blud0 = blud0[random_index, :]
                    # Change bluds shape to rectangular for ease of visualisation
                    occ_shape = factor_int(blud0.shape[0])
                    print(f'occ shape is {occ_shape}')
                    blud0_grid = torchvision.utils.make_grid(torch.abs(torch.reshape(blud0, occ_shape)))
                    oc_attributions_grid0 = torchvision.utils.make_grid(torch.abs(oc_attributions0))

                    # Write to tensorboard
                    # Bluds
                    if occlusion_count == 0:
                        writer.add_image('Interpretability/Image', image_normaliser(image_grid), running_iter)
                        # writer.add_image('Interpretability/Attributions', image_normaliser(attributions_grid), running_iter)
                        writer.add_image('Interpretability/OC_Image', image_normaliser(oc_attributions_grid0),
                                         running_iter)
                        writer.add_image('Interpretability/Bloods', image_normaliser(blud0_grid), running_iter)
                    occlusion_count += 1
                running_mvp_features.append(mvp_features)

            # Save model
            if SAVE and append_string == 'best':
                MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
                print(MODEL_PATH)
                # if epoch != (EPOCHS - 1):
                #     torch.save({'model_state_dict': model.state_dict(),
                #                 'optimizer_state_dict': optimizer.state_dict(),
                #                 'scheduler_state_dict': scheduler.state_dict(),
                #                 'epoch': epoch,
                #                 'loss': loss,
                #                 'running_iter': running_iter,
                #                 'batch_size': bs,
                #                 'resolution': input_size}, MODEL_PATH)
                # elif epoch == (EPOCHS - 1):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'loss': loss,
                            'running_iter': running_iter,
                            'batch_size': bs,
                            'resolution': input_size,
                            'running_val_preds': running_val_preds,
                            'running_val_labels': running_val_labels,
                            'running_val_names': running_val_names,
                            'running_val_mse': running_val_mse,
                            'running_val_mae': running_val_mae,
                            'overall_val_preds': overall_val_preds,
                            'overall_val_labels': overall_val_labels,
                            'overall_val_names': overall_val_names,
                            'overall_val_mse': overall_val_mse,
                            'overall_val_mae': overall_val_mae,
                            'running_mvp_features': running_mvp_features,
                            'overall_mvp_features': overall_mvp_features}, MODEL_PATH)

            if best_counter >= 5:
                # Set overalls to best epoch
                best_epoch = int(np.argmin(running_val_mse))
                print(f'The best epoch is Epoch {best_epoch}')
                overall_val_mse.append(running_val_mse[best_epoch])
                overall_val_mae.append(running_val_mae[best_epoch])
                overall_val_names.extend(running_val_names[best_epoch])
                overall_val_preds.extend(running_val_preds[best_epoch])
                overall_val_labels.extend(running_val_labels[best_epoch])
                overall_mvp_features.extend(running_mvp_features[-1])
                break

    # Now that this fold's training has ended, want starting points of next fold to reset
    latest_epoch = -1
    latest_fold = 0
    running_iter = 0

    # Print various fold outputs: Sanity check
    # print(f'Fold {fold} val_preds: {val_preds}')
    # print(f'Fold {fold} val_labels: {val_labels}')
    # print(f'Fold {fold} overall_val_mse: {overall_val_mse}')
    # print(f'Fold {fold} overall_val_mae: {overall_val_mae}')


## Totals
overall_val_labels = np.array(overall_val_labels)
overall_val_preds = np.array(overall_val_preds)

overall_val_mse = np.array(overall_val_mse)
overall_val_mae = np.array(overall_val_mae)

# Folds analysis
print('Labels', len(overall_val_labels), 'Preds', len(overall_val_preds), 'MSEs', len(overall_val_mse), 'MSEs', len(overall_val_mae))
correct = ((overall_val_preds - overall_val_labels)**2).sum()
acc = correct / len(overall_val_labels)

# Folds MSEs
folds_mse = calc_mse(overall_val_labels, overall_val_preds)
folds_mae = calc_mae(overall_val_labels, overall_val_preds)
print('MSE mean:', np.mean(overall_val_mse), 'std:', np.std(overall_val_mse))
print('MAE mean:', np.mean(overall_val_mae), 'std:', np.std(overall_val_mae))

print(f'Length of overall_val_names, overall_val_labels, overall_val_preds, overall_mvp_features are {len(overall_val_names)},'
      f'{len(overall_val_labels.tolist())}, {len(overall_val_preds.tolist())}, {len(overall_mvp_features)}')
sub = pd.DataFrame({"Filename": overall_val_names, "Inv_Time_To_Death": overall_val_labels.tolist(), "Pred": overall_val_preds.tolist(),
                    "MVP_feat": overall_mvp_features})
sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

## Plot
# Save relevant data to csv
overall_val_labels = [x[0] for x in overall_val_labels]
overall_val_preds = [x[0] for x in overall_val_preds]
sub = pd.DataFrame({"Filename": overall_val_names, "Inv_Time_To_Death": overall_val_labels, "Pred": overall_val_preds, "MVP_feat": overall_mvp_features})
sub_name = f'preds-bs{bs}-logreg-{arguments.job_name}-mse-{folds_mse}-mae-{folds_mae}.csv'
sub.to_csv(os.path.join(SAVE_PATH, sub_name), index=False)

print('END')
