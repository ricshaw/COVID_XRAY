import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_curve
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
import sys
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import balanced_accuracy_score as accuracy_score

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
    Occlusion,
    FeatureAblation,
    Saliency
)
from captum.attr import visualization as viz
import cv2
from skimage import color
import matplotlib
import time


# Function for proper handling of bools in argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--labels', nargs='+', type=str)
parser.add_argument('--images_dir', nargs='+', type=str)
parser.add_argument('--job_name', type=str)
parser.add_argument('--resolution', type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--latest_flag', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--multi_flag', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--images_flag', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--bloods_flag', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--occ_flag', type=str2bool, nargs='?', const=True, default=False)
arguments = parser.parse_args()

# Latest + multi flags
latest_flag = arguments.latest_flag
multi_flag = arguments.multi_flag
images_flag = arguments.images_flag
bloods_flag = arguments.bloods_flag
occ_flag = arguments.occ_flag
if arguments.mode == 'train':
    do_train = True
elif arguments.mode == 'test':
    do_train = False

print(f'The flags are: \nLatest: {latest_flag}\nMulti: {multi_flag}\nImages: {images_flag}\nBloods: {bloods_flag}')

# Writer will output to ./runs/ directory by default
log_dir = f'/nfs/home/pedro/COVID/logs/{arguments.job_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# writer = SummaryWriter(log_dir=log_dir)


# Figures dir
fig_dir = f'/nfs/home/pedro/COVID/Figures/{arguments.job_name}'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def get_latest(some_df):
    some_df['CXR_datetime'] = pd.to_datetime(some_df.CXR_datetime, dayfirst=True)
    some_df = some_df.groupby('patient_pseudo_id').apply(pd.DataFrame.sort_values, 'CXR_datetime',
                                               ascending=False).reset_index(drop=True)
    return some_df.drop_duplicates(subset='patient_pseudo_id', keep='first').reset_index(drop=True)


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img


def image_normaliser(some_image):
    return 1 * (some_image - torch.min(some_image)) / (torch.max(some_image) - torch.min(some_image))


def get_feats(df, i, aug=False):
    age = df.Age[i].astype(np.float32)
    gender = df.Gender[i].astype(np.float32)
    ethnicity = df.Ethnicity[i].astype(np.float32)
    bloods = df.loc[i, bloods_cols].values.astype(np.float32)
    if aug:
        bloods += np.random.normal(0, 0.2, bloods.shape)
    feats = np.concatenate((bloods, [age, gender, ethnicity]), axis=0)
    return feats


def get_feats_noi(df, aug=False):
    age = np.array(df.Age.astype(np.float32))[..., None]
    gender = np.array(df.Gender.astype(np.float32))[..., None]
    ethnicity = np.array(df.Ethnicity.astype(np.float32))[..., None]
    bloods = df.loc[:, bloods_cols].values.astype(np.float32)
    if aug:
        bloods += np.random.normal(0, 0.2, bloods.shape)
    feats = np.concatenate((bloods, np.concatenate((age, gender, ethnicity), axis=1)), axis=1)
    return feats


def get_image(df, i, transform, A_transform=None):
    image = default_image_loader(df.Filename[i])
    # Centre crop
    image = transforms.CenterCrop(min(image.size))(image)
    # A transform
    if A_transform is not None:
        image = np.array(image)
        image = A_transform(image=image)['image']
        image = Image.fromarray(image)
    # Transform
    image = transform(image)
    return image


class ImageDataset(Dataset):
    def __init__(self, df, transform, A_transform=None):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform

    def __getitem__(self, index):
        # Instantiate in case flag is false
        image, feats = np.array([]), np.array([])
        if 'Filename' in self.df.columns:
            filepath = self.df.Filename[index]
        else:
            filepath = self.df['patient_pseudo_id'][index]
        # Images
        if images_flag:
            image = get_image(self.df, index, self.transform, self.A_transform)
        # Features
        if bloods_flag:
            feats = get_feats(self.df, index, aug=True)
            time = self.df.rel_datetime[index].astype(np.float32)
            feats = np.append(feats, time)
        # Label
        label = self.df.Died[index]
        return image, filepath, label, feats

    def __len__(self):
        return self.df.shape[0]


class MultiImageDataset(Dataset):
    def __init__(self, df, transform, A_transform=None):
        self.df = df
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform

    def get_image_features(self, patient_entry, days_diff, include_images=False, include_features=False):
        # Instantiate in case flags are false
        image = np.array([])
        features = np.array([])
        filepath = patient_entry.Filename.tolist()[0]
        if include_images:
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

        label = patient_entry['Died']
        if include_features:
            features = get_feats_noi(patient_entry, aug=True)
            time = days_diff
            features = np.append(features, time)
        return image, filepath, label, features

    def __getitem__(self, index):
        # Instantiate in case flags are false
        # Create dataframe of uniques
        # unique_df = self.df.drop_duplicates(subset='patient_pseudo_id', keep='first')
        patient_id = self.df.patient_pseudo_id[index]
        df_patient = self.df[self.df.patient_pseudo_id == patient_id]
        # Randomly select two rows
        if len(df_patient) > 1:
            random_entries = df_patient.sample(n=2, replace=False)

            # Find which one is the earliest
            scan_time_one = random_entries.iloc[0].CXR_datetime
            # print(f'Scan time one is {scan_time_one}')
            scan_time_two = random_entries.iloc[1].CXR_datetime

            scan_date_one = scan_time_one
            scan_date_two = scan_time_two
            # scan_date_one = datetime.datetime(year=int(scan_time_one[6:10]),
            #                                   month=int(scan_time_one[3:5]),
            #                                   day=int(scan_time_one[0:2]),
            #                                   hour=int(scan_time_one[11:13]),
            #                                   minute=int(scan_time_one[14:16]))
            #
            # scan_date_two = datetime.datetime(year=int(scan_time_two[6:10]),
            #                                   month=int(scan_time_two[3:5]),
            #                                   day=int(scan_time_two[0:2]),
            #                                   hour=int(scan_time_two[11:13]),
            #                                   minute=int(scan_time_two[14:16]))

            if scan_date_one > scan_date_two:
                # scan date one is later
                early_entry = random_entries.iloc[[1]]
                later_entry = random_entries.iloc[[0]]
            elif scan_date_one < scan_date_two:
                # scan date one is earlier
                early_entry = random_entries.iloc[[0]]
                later_entry = random_entries.iloc[[1]]
            else:
                early_entry = random_entries.iloc[[0]]
                later_entry = early_entry

            # Calculate time difference in days
            time_diff = scan_date_two - scan_date_one
            # Convert to days
            timestamp = np.abs(time_diff.total_seconds() / datetime.timedelta(days=1).total_seconds())
            timestamp = np.array([timestamp])
            # if images_flag:
            early_image, early_filepath, label, early_features = self.get_image_features(early_entry, timestamp,
                                                                                         include_images=images_flag,
                                                                                         include_features=bloods_flag)
            later_image, later_filepath, _, later_features = self.get_image_features(later_entry, timestamp,
                                                                                     include_images=images_flag,
                                                                                     include_features=bloods_flag)
        else:
            # Duplicate entries
            random_entries = df_patient.sample(n=2, replace=True)

            # Order irrelevant
            early_entry = random_entries.iloc[[1]]
            later_entry = random_entries.iloc[[0]]

            timestamp = 0.0

            early_image, early_filepath, label, early_features = self.get_image_features(early_entry, timestamp,
                                                                                         include_images=images_flag,
                                                                                         include_features=bloods_flag)
            later_image, later_filepath, label, later_features = self.get_image_features(later_entry, timestamp,
                                                                                         include_images=images_flag,
                                                                                         include_features=bloods_flag)
        return np.array(early_image), np.array(later_image), early_filepath, later_filepath,\
               label.to_list()[0], np.array(early_features), np.array(later_features), index

    def __len__(self):
        return self.df.shape[0]


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
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLossMulti(nn.Module):
    def __init__(self, alpha, gamma=2, reduce=True):
        super(FocalLossMulti, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        probs = inputs.softmax(dim=1)
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p if t > 0 else 1-p
        weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # w = alpha if t > 0 else 1-alpha
        weight = weight * (1 - pt).pow(self.gamma)
        weight = weight.detach()
        F_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight, reduce=False)
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLossMultiFB(nn.Module):
    def __init__(self, alpha, gamma=2, reduce=True):
        super(FocalLossMultiFB, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        p = torch.softmax(inputs, dim=1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        p_t = p * targets + (1 - p) * (1 - targets)
        F_loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * F_loss

        if self.reduce:
            F_loss = F_loss.mean()
            return F_loss
        else:
            return F_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


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


def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# Function to pad empty lists in case occlusion is absent
def pad_dict_list(dict_list, padel=0):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


# Function to address processing needed to save array of shape (BS x 3 x N x N)
def reshape_array(input_array, colormap_grid=False):
    print(f'The minimum and maximum attributions are {input_array.min()}, {input_array.max()}')
    array_length = input_array.shape[-1]
    batch_length = input_array.shape[0]
    if colormap_grid:  # Implies Occlusion grid
        reshaped_grid = np.zeros((array_length * batch_length, array_length))
        for batch in range(batch_length):
            reshaped_grid[array_length * batch:array_length * (batch + 1), :] = input_array[batch, 0, ...]
        cmapper = matplotlib.cm.get_cmap('hot')
        reshaped_grid = cmapper(reshaped_grid)
    else:
        reshaped_grid = np.zeros((3, array_length * batch_length, array_length))
        for batch in range(batch_length):
            reshaped_grid[:, array_length * batch:array_length * (batch + 1), :] = input_array[batch, ...]
        reshaped_grid = np.transpose(reshaped_grid, [1, 2, 0])
        reshaped_grid = color.rgb2gray(reshaped_grid)
        reshaped_grid -= reshaped_grid.min()
    reshaped_grid /= reshaped_grid.max()
    reshaped_grid = Image.fromarray(np.int8(255 * reshaped_grid))
    return reshaped_grid


# Some necessary variables
labels = arguments.labels  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
test_labels = '/nfs/home/pedro/COVID/Labels/gstt.csv'  # Load test data
test_df = pd.read_csv(test_labels)
if latest_flag:
    test_df = get_latest(test_df)

# Remove entries with no Filename
test_df = test_df.dropna(subset=['Filename'])
# Remove entries with too great a mismatch
tester = test_df[test_df.Time_Mismatch < 2]
test_df['Filename'] = '/nfs/project/covid/CXR/GSTT/GSTT_JPGs_All' + '/' + test_df['Filename'].astype(str)

print('Test data:', test_df.shape)
tmp = test_df.drop(columns=['patient_pseudo_id', 'CXR_datetime', 'Age', 'Gender', 'Ethnicity', 'Died', 'Filename', 'Time_Mismatch'])
bloods_cols = tmp.columns
print('Bloods:', bloods_cols)

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
    EPOCHS = 100
    FOLDS = 5
    input_size = (arguments.resolution, arguments.resolution)
    encoder = 'efficientnet-b3'
else:
    running_iter = 0
    loaded_epoch = -1
    if arguments.resolution > 100 and images_flag:
        bs = 32
    else:
        bs = 128
    EPOCHS = 100
    FOLDS = 5
    input_size = (arguments.resolution, arguments.resolution)
    encoder = 'efficientnet-b3'

# Load labels
print(f'The  labels are {labels}')
print(f'The batch size is {bs}')
if len(labels) == 1:
    labels = labels[0]
    df = pd.read_csv(labels)
    # Time point analysis


    df['Filename'] = arguments.images_dir[0] + '/' + df['Filename'].astype(str)
    ## Load train data
    if latest_flag:
        df = get_latest(df)

elif len(labels) > 1:
    df = pd.read_csv(labels[0])
    for extra in range(1, len(labels)):
        extra_df = pd.read_csv(labels[extra])
        df = pd.concat([df, extra_df], ignore_index=True)


## Replace data
def prepare_data(df, bloods_cols, myscaler=None):
    print('Preparing data')
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
    df['rel_datetime'] = (df['CXR_datetime'] - df['min_datetime']) / np.timedelta64(1, 'D')

    # Extract features
    bloods = df.loc[:, bloods_cols].values.astype(np.float32)
    # Bloods is a numpy array
    print('Bloods', bloods.shape)
    age = df.Age[:, None]
    gender = df.Gender[:, None]
    ethnicity = df.Ethnicity[:, None]
    time = df.rel_datetime[:, None]

    # Normalise features
    if not myscaler:
        scaler = StandardScaler()
        X = np.concatenate((bloods, age, gender, ethnicity, time), axis=1)
        scaler.fit(X)
        X = scaler.transform(X)
    else:
        X = np.concatenate((bloods, age, gender, ethnicity, time), axis=1)
        myscaler.fit(X)
        X = myscaler.transform(X)
    # Fill missing
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(X)
    X = imputer.transform(X)

    # Put back features
    print(f'df cols are {df.columns}')
    df.loc[:, bloods_cols] = X[:, 0:bloods.shape[1]]
    df.loc[:, 'Age'] = X[:, bloods.shape[1]]
    df.loc[:, 'Gender'] = X[:, bloods.shape[1]+1]
    df.loc[:, 'Ethnicity'] = X[:, bloods.shape[1]+2]
    df.loc[:, 'rel_datetime'] = X[:, bloods.shape[1]+3]
    if not myscaler:
        return df, scaler
    else:
        return df


# For shape purposes:
# first_blood = '.cLac'
# last_blood = 'OBS BMI Calculation'
# bloods = df.loc[:, first_blood:last_blood]
# first_vital = 'Fever (finding)'
# last_vital = 'Immunodeficiency disorder (disorder)'
# vitals = df.loc[:, first_vital:last_vital]
# age = df.Age
# gender = df.Gender
# ethnicity = df.Ethnicity
# days_from_onset_to_scan = df['days_from_onset_to_scan']
# temp_bloods = pd.concat([bloods, age, gender, ethnicity, days_from_onset_to_scan, vitals], axis=1, sort=False)
temp_bloods_columns = bloods_cols.to_list()
temp_bloods_columns.extend(['Age', 'Gender', 'Ethnicity', 'rel_datetime'])
temp_bloods = pd.DataFrame(columns=temp_bloods_columns)
print(bloods_cols.to_list())
print(temp_bloods)
print(f'Length of temp bloods is {len(temp_bloods.columns)}')

# # Exclude all entries with "Missing" Died stats
# df = df[~df['Died'].isin(['Missing'])]
# df['Died'] = pd.to_numeric(df['Died'])

# Augmentations
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         n_feats = len(temp_bloods.columns)
#         hidden1 = 256
#         hidden2 = 256
#         dropout = 0.3
#         self.fc1 = nn.Linear(n_feats, hidden1, bias=True)
#         self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
#         self.meta = nn.Sequential(self.fc1,
#                                   # nn.BatchNorm1d(hidden1),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout),
#                                   self.fc2,
#                                   # nn.BatchNorm1d(hidden2),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout)
#                                   )
#
#         self.classifier = nn.Linear(hidden2, out_features=1, bias=True)
#
#     def forward(self, features):
#         features = self.meta(features)
#         out = self.classifier(features)
#         # out = self.net(x)
#         return out


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
    def __init__(self, encoder='efficientnet-b0', nfeats=33):
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
        if images_flag:
            self.net = EfficientNet.from_pretrained(encoder)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.out_chns += n_channels_dict[encoder]
        if bloods_flag:
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
        if multi_flag:
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
        if multi_flag:
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


use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')

# For aggregation
overall_val_preds = []
overall_val_labels = []
overall_val_names = []

overall_val_roc_aucs = []
overall_val_pr_aucs = []
overall_mvp_features = []

overall_cont_features = []
overall_ig_mvp_features = []
overall_ig_cont_features = []
overall_ignt_mvp_features = []
overall_ignt_cont_features = []
# overall_dl_mvp_features = []
# overall_dl_cont_features = []
overall_fa_mvp_features = []
overall_fa_cont_features = []

overall_mvp_features1 = []

overall_cont_features1 = []
overall_ig_mvp_features1 = []
overall_ig_cont_features1 = []
overall_ignt_mvp_features1 = []
overall_ignt_cont_features1 = []
# overall_dl_mvp_features1 = []
# overall_dl_cont_features1 = []
overall_fa_mvp_features1 = []
overall_fa_cont_features1 = []

# Occlusion
x_shape = 2
x_stride = 2

alpha = 0.75
gamma = 2.0
CUTMIX_PROB = 1.0
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

# Model wrappers for occlusion
if multi_flag:
    if not images_flag:
        def model_wrapper(vec1, vec2):
            default_im = torch.FloatTensor()
            return model(default_im, vec1, default_im, vec2)
    elif not bloods_flag:
        def model_wrapper(im1, im2):
            default_vec = torch.FloatTensor()
            return model(im1, default_vec, im2, default_vec)
else:
    if not images_flag:
        def model_wrapper(vec1):
            default_all = torch.FloatTensor()
            return model(default_all, vec1, default_all, default_all)
    elif not bloods_flag:
        def model_wrapper(im1):
            default_all = torch.FloatTensor()
            return model(im1, default_all, default_all, default_all)
    else:
        # When both are present still need a wrapper to pass one of each input type
        def model_wrapper(im1, vec1):
            default_all = torch.FloatTensor()
            return model(im1, vec1, default_all, default_all)

if do_train:
    # Best epoch tracking
    best_epochs = []

    for fold in range(latest_fold, FOLDS):
        print('\nFOLD', fold)
        # Running lists
        running_val_preds = []
        running_val_labels = []
        running_val_names = []
        running_val_roc_aucs = []
        running_val_pr_aucs = []
        running_mvp_features = []

        running_cont_features = []
        running_ig_mvp_features = []
        running_ig_cont_features = []
        running_ignt_mvp_features = []
        running_ignt_cont_features = []
        # running_dl_mvp_features = []
        # running_dl_cont_features = []
        running_fa_mvp_features = []
        running_fa_cont_features = []

        running_mvp_features1 = []

        running_cont_features1 = []
        running_ig_mvp_features1 = []
        running_ig_cont_features1 = []
        running_ignt_mvp_features1 = []
        running_ignt_cont_features1 = []
        # running_dl_mvp_features1 = []
        # running_dl_cont_features1 = []
        running_fa_mvp_features1 = []
        running_fa_cont_features1 = []

        # Pre-loading sequence
        # model = Model()
        ## Init model
        singlemodel = SingleModel(encoder).cuda()
        model = CombinedModel(singlemodel).cuda()
        model = nn.DataParallel(model)

        optimizer = RangerLars(model.parameters())

        # alpha = torch.FloatTensor([0.9, 0.8, 0.7, 0.25])[None, ...].cuda()
        # criterion = FocalLoss(logits=True)
        # optimizer = RangerLars(model.parameters())
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
            try:
                loaded_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
                checkpoint = torch.load(os.path.join(SAVE_PATH, loaded_model_file), map_location=torch.device('cuda:0'))
            except FileNotFoundError:
                loaded_model_file = f'model_epoch_0_fold_{fold}.pth'
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
            running_val_roc_aucs = checkpoint['running_val_roc_aucs']
            running_val_pr_aucs = checkpoint['running_val_pr_aucs']
            overall_val_preds = checkpoint['overall_val_preds']
            overall_val_labels = checkpoint['overall_val_labels']
            overall_val_names = checkpoint['overall_val_names']
            overall_val_roc_aucs = checkpoint['overall_val_roc_aucs']
            overall_val_pr_aucs = checkpoint['overall_val_pr_aucs']
            running_mvp_features = checkpoint['running_mvp_features']
            overall_mvp_features = checkpoint['overall_mvp_features']
            best_epochs = checkpoint['best_epochs']
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

        train_df, train_scaler = prepare_data(train_df, bloods_cols)
        train_df = train_df.reset_index(drop=True, inplace=False)
        print(f'The df columns are {train_df.columns}')
        val_df, _ = prepare_data(val_df, bloods_cols)
        val_df = val_df.reset_index(drop=True, inplace=False)
        if multi_flag:
            train_dataset = MultiImageDataset(train_df, train_transform, A_transform)
        else:
            train_dataset = ImageDataset(train_df, train_transform, A_transform)
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)

        val_dataset = ImageDataset(val_df, tta_transform)
        val_loader = DataLoader(val_dataset, batch_size=int(bs/4), num_workers=8, shuffle=False)

        int_dataset = ImageDataset(val_df, val_transform)
        int_loader = DataLoader(int_dataset, batch_size=bs, num_workers=8, shuffle=False)

        print(f'The shape of the labels are: {len(int_loader), len(val_loader), len(train_loader)}')
        # for colu in df.columns:
        #     print(colu)
        # Best model selection
        best_val_auc = 0.0
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
                    # Loading depends on multi or not
                    if multi_flag:
                        early_images, later_images, names, _, labels, early_bloods, later_bloods, idx = sample[0], sample[1], \
                                                                                                        sample[2], sample[3], \
                                                                                                        sample[4], sample[5], \
                                                                                                        sample[6], sample[7]
                        print(early_images.shape, later_images.shape, early_bloods.shape, later_bloods.shape)
                        # First set
                        early_images = early_images.cuda()
                        early_bloods = early_bloods.cuda()
                        early_bloods = early_bloods.float()
                        # Second set
                        later_images = later_images.cuda()
                        later_bloods = later_bloods.cuda()
                        later_bloods = later_bloods.float()
                        # Unchanged
                        labels = labels.cuda()
                        labels = labels.unsqueeze(1).float()
                    else:
                        early_images, names, labels, early_bloods = sample[0], sample[1], sample[2], sample[3]
                        later_images, later_bloods = None, None
                        early_images = early_images.cuda()
                        labels = labels.cuda()
                        labels = labels.unsqueeze(1).float()
                        early_bloods = early_bloods.cuda()
                        early_bloods = early_bloods.float()

                        # print(early_images.shape)
                        # print(early_bloods.shape)
                        # print(labels.shape)

                    prob = np.random.rand(1)
                    if prob < CUTMIX_PROB:
                        # generate mixed sample
                        lam = np.random.beta(1, 1)
                        rand_index = torch.randperm(labels.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        features_a = early_bloods
                        features_b = early_bloods[rand_index]
                        features = features_a * lam + features_b * (1. - lam)
                        # compute output
                        # print('HERE', early_images.nelement(), early_images.shape)
                        out = model(early_images, early_bloods, later_images, later_bloods)
                        # loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
                        loss = sigmoid_focal_loss(out, target_a, alpha, gamma, reduction="mean") * lam + \
                               sigmoid_focal_loss(out, target_b, alpha, gamma, reduction="mean") * (1. - lam)

                    else:
                        out = model(early_images, early_bloods, later_images, later_bloods)
                        # loss = criterion(out, labels)
                        loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

                    out = torch.sigmoid(out)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    total += labels.numel()

                    train_acc += ((out > 0.5).int() == labels).sum().item()
                    # out = torch.sigmoid(out)
                    # correct += ((out > 0.5).int() == labels).sum().item()

                    # Name check: Shuffling sanity check
                    if i == 0:
                        print(f'The test names are: {names[0]}, {names[-2]}')

                    # Convert labels and output to grid
                    labels_grid = torchvision.utils.make_grid(labels)
                    rounded_output_grid = torchvision.utils.make_grid((out > 0.5).int())
                    output_grid = torchvision.utils.make_grid(out)

                    # Writing to tensorboard
                    if running_iter % 50 == 0:
                        writer.add_scalar('Loss/train', loss.item(), running_iter)
                        writer.add_image('Visuals/Labels', image_normaliser(labels_grid), running_iter)
                        writer.add_image('Visuals/Output', image_normaliser(output_grid), running_iter)
                        # writer.add_image('Visuals/Rounded Output', image_normaliser(rounded_output_grid), running_iter)

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
                    if multi_flag:
                        for pid in val_df['patient_pseudo_id'].unique():
                            # print(pid)
                            pid_df = val_df[val_df['patient_pseudo_id'] == pid].reset_index(drop=True)
                            pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                            pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                            n_images = pid_df.shape[0]
                            # print(n_images, 'images')
                            outs = []
                            labs = []
                            for n in range(n_images):
                                ind1 = 0
                                ind2 = n
                                time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                                time2 = pid_df.rel_datetime[ind2].astype(np.float32)

                                image1, image2 = np.array([]), np.array([])
                                feats1, feats2 = np.array([]), np.array([])
                                if bloods_flag:
                                    # Features
                                    feats1 = get_feats(pid_df, ind1, aug=False)
                                    feats2 = get_feats(pid_df, ind2, aug=False)
                                    feats1 = np.append(feats1, time1)
                                    feats2 = np.append(feats2, time2)
                                else:
                                    feats1, feats2 = np.array([]), np.array([])
                                if images_flag:
                                    # Image
                                    image1 = get_image(pid_df, ind1, tta_transform)
                                    image2 = get_image(pid_df, ind2, tta_transform)
                                else:
                                    # image1, image2 = torch.FloatTensor(), torch.FloatTensor()
                                    image1, image2 = torch.FloatTensor(), torch.FloatTensor()

                                # Label
                                labels = pid_df.Died[ind1]
                                labels = torch.Tensor([labels]).cuda()
                                labels = labels.unsqueeze(1).float()

                                # Features
                                feats1 = torch.Tensor(feats1).cuda()
                                feats1 = feats1.unsqueeze(0)
                                feats2 = torch.Tensor(feats2).cuda()
                                feats2 = feats2.unsqueeze(0)

                                # Images
                                image1, image2 = image1.cuda(), image2.cuda()
                                image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)

                                ## TTA
                                if len(image1.size()) == 5:
                                    batch_size, n_crops, c, h, w = image1.size()
                                    image1 = image1.view(-1, c, h, w)
                                    image2 = image2.view(-1, c, h, w)
                                    if bloods_flag:
                                        _, n_feats = feats1.size()
                                        feats1 = feats1.repeat(1, n_crops).view(-1, n_feats)
                                        feats2 = feats2.repeat(1, n_crops).view(-1, n_feats)
                                    out = model(image1, feats1, image2, feats2)
                                    out = out.view(batch_size, n_crops, -1).mean(1)
                                else:
                                    out = model(image1, feats1, image2, feats2)
                                out = torch.sigmoid(out)
                                outs += out.cpu().numpy().tolist()
                                labs += labels.cpu().numpy().tolist()
                            outs_mean = np.mean(outs)
                            val_preds += outs  #.cpu().numpy().tolist()
                            val_labels += labs
                            val_names += [pid] * n_images
                            total += labels.numel()
                            acc = ((np.array(outs) > 0.5).astype(int) == labs).sum().item()
                            # acc = ((outs > 0.5).int() == labs).sum().item()
                            val_counter += 1
                    else:
                        for i, sample in enumerate(val_loader):
                            early_images, names, labels, early_bloods = sample[0], sample[1], sample[2], sample[3]
                            # print(f'Val images shape is {early_images.shape}')
                            later_images, later_bloods = None, None

                            labels = labels.cuda()
                            labels = labels.unsqueeze(1).float()
                            early_bloods = early_bloods.cuda()
                            early_bloods = early_bloods.float()

                            # TTA
                            if len(early_images.size()) == 5:
                                batch_size, n_crops, c, h, w = early_images.size()
                                early_images = early_images.view(-1, c, h, w)
                                if bloods_flag:
                                    _, n_feats = early_bloods.size()
                                    early_bloods = early_bloods.repeat(1, n_crops).view(-1, n_feats)
                                out = model(early_images, early_bloods, later_images, later_bloods)
                                out = out.view(batch_size, n_crops, -1).mean(1)
                            else:
                                out = model(early_images, early_bloods, later_images, later_bloods)

                            val_loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")
                            out = torch.sigmoid(out)

                            running_loss += val_loss.item()

                            total += labels.numel()
                            # out = torch.sigmoid(out)

                            # Save validation output for post all folds training aggregation
                            val_preds += out.cpu().numpy().tolist()
                            val_labels += labels.cpu().numpy().tolist()
                            val_names += names

                            acc = ((out > 0.5).int() == labels).sum().item()
                            # correct += ((out > 0.5).int() == labels).sum().item()
                            val_counter += 1

                # Write to tensorboard
                writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)

                # acc = correct / total
                acc = ((out > 0.5).int() == labels).sum().item()
                val_acc = acc / total
                y_true = np.array(val_labels)
                y_scores = np.array(val_preds)

                # Overalls
                true_auc = roc_auc_score(y_true, y_scores)
                precision_overall, recall_overall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
                true_pr_auc = auc(recall_overall, precision_overall)

                # Aggregation
                print(f'The val lengths are: {len(val_names), len(val_preds), len(val_labels), i, len(val_df)}')
                running_val_names.append(val_names)
                running_val_preds.append(val_preds)
                running_val_labels.append(val_labels)
                running_val_roc_aucs.append(true_auc)
                running_val_pr_aucs.append(true_pr_auc)
                print("Epoch: {}, Loss: {},\n Test Accuracy: {},\n ROC-AUCs: {},\n PR-AUCs {}\n".format(epoch,
                                                                                                        running_loss,
                                                                                                        val_acc,
                                                                                                        true_auc,
                                                                                                        true_pr_auc))
                writer.add_scalar('Loss/AUC', true_auc, running_iter)
                writer.add_scalar('Loss/PR_AUC', true_pr_auc, running_iter)

                # Check if better than current best:
                if true_auc > best_val_auc:
                    best_val_auc = true_auc
                    append_string = 'best'
                    best_counter = 0
                else:
                    append_string = 'nb'
                    best_counter += 1

                if append_string == 'best':  # and epoch > 5:
                # if (epoch == (EPOCHS - 1)) or (epoch % 10 == 0):
                    occlusion = occ_flag
                else:
                    occlusion = False
                if occlusion:
                    # Some variables
                    print(f'Running occlusion on fold {fold}!')
                    mvp_features = []
                    ig_mvp_features = []
                    ignt_mvp_features = []
                    # dl_mvp_features = []
                    fa_mvp_features = []
                    cont_features = []
                    ig_cont_features = []
                    ignt_cont_features = []
                    # dl_cont_features = []
                    fa_cont_features = []

                    mvp_features1 = []
                    ig_mvp_features1 = []
                    ignt_mvp_features1 = []
                    # dl_mvp_features1 = []
                    fa_mvp_features1 = []
                    cont_features1 = []
                    ig_cont_features1 = []
                    ignt_cont_features1 = []
                    # dl_cont_features1 = []
                    fa_cont_features1 = []
                    occlusion_count = 0
                    death_image_flag = 2
                    survival_image_flag = 2

                    print(len(int_loader))

                    if multi_flag:
                        print('Multi validation')
                        for pid in val_df['patient_pseudo_id'].unique():
                            # print(pid)
                            pid_df = val_df[val_df['patient_pseudo_id'] == pid].reset_index(drop=True)
                            pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                            pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                            n_images = pid_df.shape[0]
                            # print(n_images, 'images')
                            for n in range(n_images):
                                ind1 = 0
                                ind2 = n
                                time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                                time2 = pid_df.rel_datetime[ind2].astype(np.float32)

                                image1, image2 = np.array([]), np.array([])
                                feats1, feats2 = np.array([]), np.array([])
                                if bloods_flag:
                                    # Features
                                    feats1 = get_feats(pid_df, ind1, aug=False)
                                    feats2 = get_feats(pid_df, ind2, aug=False)
                                    feats1 = np.append(feats1, time1)
                                    feats2 = np.append(feats2, time2)
                                else:
                                    feats1, feats2 = np.array([]), np.array([])
                                if images_flag:
                                    # Image
                                    image1 = get_image(pid_df, ind1, tta_transform)
                                    image2 = get_image(pid_df, ind2, tta_transform)
                                else:
                                    image1, image2 = torch.FloatTensor(), torch.FloatTensor()

                                # Label
                                labels = pid_df.Died[ind1]
                                labels = torch.Tensor([labels]).cuda()
                                labels = labels.unsqueeze(1).float()

                                # Features
                                feats1 = torch.Tensor(feats1).cuda()
                                feats1 = feats1.unsqueeze(0)
                                feats2 = torch.Tensor(feats2).cuda()
                                feats2 = feats2.unsqueeze(0)

                                # Images
                                image1, image2 = image1.cuda(), image2.cuda()
                                image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)

                                # Always create
                                baseline_bloods = torch.zeros_like(feats1).cuda().float()
                                baseline = torch.zeros_like(image1).cuda()

                                # Int for bloods only
                                if bloods_flag and not images_flag:
                                    # Calculate attribution scores + delta
                                    oc = Occlusion(model_wrapper)
                                    # ig = IntegratedGradients(model)
                                    ig = IntegratedGradients(model_wrapper)
                                    # dl = DeepLift(model)
                                    # gs = GradientShap(model)
                                    fa = FeatureAblation(model_wrapper)
                                    ig_nt = NoiseTunnel(ig)
                                    # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                    start_oc = time.time()
                                    blud0, blud1 = oc.attribute((feats1, feats2), sliding_window_shapes=((1,), (1,)),
                                                                strides=((1,), (1,)), target=None,
                                                                baselines=(baseline_bloods, baseline_bloods))
                                    print(f'The OC time was {time.time() - start_oc}')

                                    start_ig = time.time()
                                    ig_attr_test0, ig_attr_test1 = ig.attribute((feats1, feats2), n_steps=50)
                                    print(f'The IG time was {time.time() - start_ig}')

                                    start_ignt = time.time()
                                    ig_nt_attr_test0, ig_nt_attr_test1 = ig_nt.attribute((feats1, feats2))
                                    print(f'The IGNT time was {time.time() - start_ignt}')

                                    # start_dl = time.time()
                                    # _, dl_attr_test0, _, dl_attr_test1 = dl.attribute((image1, feats1, image2, feats2))
                                    # print(f'The DL time was {time.time() - start_dl}')
                                    # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                    # start_fa = time.time()
                                    # fa_attr_test0, fa_attr_test1 = fa.attribute((feats1, feats2))
                                    # print(f'The FA time was {time.time() - start_fa}')
                                    # print('IG + SmoothGrad Attributions:', attributions)
                                    # print('Convergence Delta:', delta)

                                    # Print
                                    for single_feature in range(blud0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        mvp_features.append(mvp_feature)
                                        cont_features.append(blud0[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ig_mvp_features.append(mvp_feature)
                                        ig_cont_features.append(ig_attr_test0[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_nt_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_nt_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ignt_mvp_features.append(mvp_feature)
                                        ignt_cont_features.append(ig_nt_attr_test0[single_feature, :].cpu().tolist())

                                    # for single_feature in range(dl_attr_test0.shape[0]):
                                    #     mvp_feature = temp_bloods.columns[
                                    #         int(np.argmax(torch.abs(dl_attr_test0[single_feature, :]).cpu()))]
                                    #     # print(f'The most valuable feature was {mvp_feature}')
                                    #     dl_mvp_features.append(mvp_feature)
                                    #     dl_cont_features.append(dl_attr_test0[single_feature, :].cpu().tolist())

                                    for single_feature in range(fa_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(fa_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        fa_mvp_features.append(mvp_feature)
                                        fa_cont_features.append(fa_attr_test0[single_feature, :].cpu().tolist())

                                    # Again, for the rest
                                    for single_feature in range(blud1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(blud1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        mvp_features1.append(mvp_feature)
                                        cont_features1.append(blud1[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ig_mvp_features1.append(mvp_feature)
                                        ig_cont_features1.append(ig_attr_test1[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_nt_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_nt_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ignt_mvp_features1.append(mvp_feature)
                                        ignt_cont_features1.append(ig_nt_attr_test1[single_feature, :].cpu().tolist())

                                    # for single_feature in range(dl_attr_test1.shape[0]):
                                    #     mvp_feature = temp_bloods.columns[
                                    #         int(np.argmax(torch.abs(dl_attr_test1[single_feature, :]).cpu()))]
                                    #     # print(f'The most valuable feature was {mvp_feature}')
                                    #     dl_mvp_features1.append(mvp_feature)
                                    #     dl_cont_features1.append(dl_attr_test1[single_feature, :].cpu().tolist())

                                    for single_feature in range(fa_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(fa_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        fa_mvp_features1.append(mvp_feature)
                                        fa_cont_features1.append(fa_attr_test1[single_feature, :].cpu().tolist())
                                elif not bloods_flag and images_flag:
                                    if occlusion_count == 0:
                                        oc = Occlusion(model_wrapper)
                                        sl = Saliency(model_wrapper)
                                        start_oc = time.time()
                                        # Images and features occlusion combined
                                        oc_attributions0, oc_attributions1 = oc.attribute((image1, image2),
                                                                                          sliding_window_shapes=(
                                                                                          (3, x_shape, x_shape),
                                                                                          (3, x_shape, x_shape)),
                                                                                          strides=((3, x_stride, x_stride),
                                                                                                   (3, x_stride, x_stride)),
                                                                                          target=None,
                                                                                          baselines=(baseline, baseline))
                                        print(f'The OC time was {time.time() - start_oc}')
                                        # Save images
                                        image_grid1 = image1.cpu().numpy()
                                        image_grid1 = reshape_array(image_grid1)
                                        cv2.imwrite(f'{fig_dir}/image_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid1))

                                        image_grid2 = image2.cpu().numpy()
                                        image_grid2 = reshape_array(image_grid2)
                                        cv2.imwrite(f'{fig_dir}/image_grid2_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid2))

                                        # Save attributions
                                        oc_attributions0 = oc_attributions0.cpu().numpy()
                                        im = reshape_array(oc_attributions0, colormap_grid=True)
                                        cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.int8(im))

                                        oc_attributions1 = oc_attributions1.cpu().numpy()
                                        im = reshape_array(oc_attributions1, colormap_grid=True)
                                        cv2.imwrite(f'{fig_dir}/oc_attributions_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.int8(im))

                                        # Saliency
                                        start_sl = time.time()
                                        sal_attributions0, sal_attributions1 = sl.attribute((image1.requires_grad_(), image2.requires_grad_()), abs=False)
                                        print(f'The Saliency time was {time.time() - start_sl}')
                                elif bloods_flag and images_flag:
                                    print(occlusion_count)
                                    if occlusion_count == 0:
                                        oc = Occlusion(model)
                                        sl = Saliency(model)
                                        start_oc = time.time()
                                        # Images and features occlusion combined
                                        oc_attributions0, blud0, oc_attributions1, blud1 = oc.attribute(
                                            (image1.requires_grad_(), feats1, image2.requires_grad_(), feats2),
                                            sliding_window_shapes=(
                                                (3, x_shape, x_shape), (1,),
                                                (3, x_shape, x_shape), (1,)),
                                            strides=((3, x_stride, x_stride), (1,),
                                                     (3, x_stride, x_stride), (1,)), target=None,
                                            baselines=(baseline, baseline_bloods, baseline, baseline_bloods))
                                        print(f'The OC time was {time.time() - start_oc}')

                                        # Save images
                                        image_grid1 = image1.cpu().numpy()
                                        image_grid1 = reshape_array(image_grid1)
                                        cv2.imwrite(f'{fig_dir}/image_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid1))

                                        image_grid2 = image2.cpu().numpy()
                                        image_grid2 = reshape_array(image_grid2)
                                        cv2.imwrite(f'{fig_dir}/image_grid2_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid2))

                                        # Save attributions
                                        oc_attributions0 = oc_attributions0.cpu().numpy()
                                        im = reshape_array(oc_attributions0, colormap_grid=True)
                                        cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.int8(im))

                                        oc_attributions1 = oc_attributions1.cpu().numpy()
                                        im = reshape_array(oc_attributions1, colormap_grid=True)
                                        cv2.imwrite(f'{fig_dir}/oc_attributions_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.int8(im))

                                        # Saliency
                                        start_sl = time.time()
                                        sal_attributions0, _, sal_attributions1, _ = sl.attribute((image1.requires_grad_(),
                                                                                                   feats1, image2.requires_grad_(),
                                                                                                   feats2), abs=False)
                                        print(f'The Saliency time was {time.time() - start_sl}')
                                    # Calculate attribution scores + delta
                                    # ig = IntegratedGradients(model)
                                    ig = IntegratedGradients(model)
                                    # dl = DeepLift(model)
                                    # gs = GradientShap(model)
                                    fa = FeatureAblation(model)
                                    ig_nt = NoiseTunnel(ig)
                                    # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                    start_oc = time.time()
                                    _, blud0, _, blud1 = oc.attribute((image1, feats1, image2, feats2),
                                                                      sliding_window_shapes=((3, image1.shape[2], image1.shape[3]), (1,), (3, image1.shape[2], image1.shape[3]), (1,)),
                                                                      strides=((3, image1.shape[2], image1.shape[3]), (1,), (3, image1.shape[2], image1.shape[3]), (1,)), target=None,
                                                                      baselines=(
                                                                      image1, baseline_bloods, image2, baseline_bloods))
                                    print(f'The OC time was {time.time() - start_oc}')

                                    start_ig = time.time()
                                    _, ig_attr_test0, _, ig_attr_test1 = ig.attribute((image1, feats1, image2, feats2),
                                                                                      n_steps=50)
                                    print(f'The IG time was {time.time() - start_ig}')

                                    start_ignt = time.time()
                                    _, ig_nt_attr_test0, _, ig_nt_attr_test1 = ig_nt.attribute((feats1, feats2))
                                    print(f'The IGNT time was {time.time() - start_ignt}')

                                    # start_dl = time.time()
                                    # _, dl_attr_test0, _, dl_attr_test1 = dl.attribute((image1, feats1, image2, feats2))
                                    # print(f'The DL time was {time.time() - start_dl}')
                                    # # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                    start_fa = time.time()
                                    _, fa_attr_test0, _, fa_attr_test1 = fa.attribute((image1, feats1, image2, feats2))
                                    print(f'The FA time was {time.time() - start_fa}')
                                    # print('IG + SmoothGrad Attributions:', attributions)
                                    # print('Convergence Delta:', delta)

                                    # Print
                                    for single_feature in range(blud0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        mvp_features.append(mvp_feature)
                                        cont_features.append(blud0[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ig_mvp_features.append(mvp_feature)
                                        ig_cont_features.append(ig_attr_test0[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_nt_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_nt_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ignt_mvp_features.append(mvp_feature)
                                        ignt_cont_features.append(ig_nt_attr_test0[single_feature, :].cpu().tolist())

                                    # for single_feature in range(dl_attr_test0.shape[0]):
                                    #     mvp_feature = temp_bloods.columns[
                                    #         int(np.argmax(torch.abs(dl_attr_test0[single_feature, :]).cpu()))]
                                    #     # print(f'The most valuable feature was {mvp_feature}')
                                    #     dl_mvp_features.append(mvp_feature)
                                    #     dl_cont_features.append(dl_attr_test0[single_feature, :].cpu().tolist())

                                    for single_feature in range(fa_attr_test0.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(fa_attr_test0[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        fa_mvp_features.append(mvp_feature)
                                        fa_cont_features.append(fa_attr_test0[single_feature, :].cpu().tolist())

                                    # Again, for the rest
                                    for single_feature in range(blud1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(blud1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        mvp_features1.append(mvp_feature)
                                        cont_features1.append(blud1[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ig_mvp_features1.append(mvp_feature)
                                        ig_cont_features1.append(ig_attr_test1[single_feature, :].cpu().tolist())

                                    for single_feature in range(ig_nt_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(ig_nt_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        ignt_mvp_features1.append(mvp_feature)
                                        ignt_cont_features1.append(ig_nt_attr_test1[single_feature, :].cpu().tolist())

                                    # for single_feature in range(dl_attr_test1.shape[0]):
                                    #     mvp_feature = temp_bloods.columns[
                                    #         int(np.argmax(torch.abs(dl_attr_test1[single_feature, :]).cpu()))]
                                    #     # print(f'The most valuable feature was {mvp_feature}')
                                    #     dl_mvp_features1.append(mvp_feature)
                                    #     dl_cont_features1.append(dl_attr_test1[single_feature, :].cpu().tolist())

                                    for single_feature in range(fa_attr_test1.shape[0]):
                                        mvp_feature = temp_bloods.columns[
                                            int(np.argmax(torch.abs(fa_attr_test1[single_feature, :]).cpu()))]
                                        # print(f'The most valuable feature was {mvp_feature}')
                                        fa_mvp_features1.append(mvp_feature)
                                        fa_cont_features1.append(fa_attr_test1[single_feature, :].cpu().tolist())
                    else:
                        print('Not Multi!')
                        for i, sample in enumerate(int_loader):
                            image1, names, labels, feats1 = sample[0], sample[1], sample[2], sample[3]
                            image2, feats2 = torch.FloatTensor(), torch.FloatTensor()
                            image1 = image1.cuda()[0:4, ...] #[None, ...]
                            labels = labels.cuda()[0:4, ...] #[None, ...]
                            labels = labels.unsqueeze(1).float()
                            feats1 = feats1.cuda()[0:4, ...] #[None, ...]
                            feats1 = feats1.float()

                            # print(f'The image shape is {image1.shape}')

                            # Account for tta: Take first image (non-augmented)
                            # Label does not need to be touched because it is obv. the same for this image regardless of tta
                            # Set a baseline
                            baseline = torch.zeros_like(image1).cuda().float()
                            baseline_bloods = torch.zeros_like(feats1).cuda().float()

                            # Int for bloods only
                            if bloods_flag and not images_flag:
                                print('Using Bloods!')
                                # Calculate attribution scores + delta
                                # ig = IntegratedGradients(model)
                                oc = Occlusion(model_wrapper)
                                ig = IntegratedGradients(model_wrapper)
                                # dl = DeepLift(model_wrapper)
                                # gs = GradientShap(model)
                                fa = FeatureAblation(model_wrapper)
                                ig_nt = NoiseTunnel(ig)
                                # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                start_oc = time.time()
                                print(image1.shape, image2.shape, feats1.shape, feats2.shape)
                                blud0 = oc.attribute(feats1,
                                                     sliding_window_shapes=(1,),
                                                     strides=(1,), target=None,
                                                     baselines=baseline_bloods)
                                print(f'The OC time was {time.time() - start_oc}')

                                start_ig = time.time()
                                ig_attr_test = ig.attribute(feats1, n_steps=50)
                                print(f'The IG time was {time.time() - start_ig}')

                                start_ignt = time.time()
                                ig_nt_attr_test = ig_nt.attribute(feats1)
                                print(f'The IGNT time was {time.time() - start_ignt}')

                                # start_dl = time.time()
                                # dl_attr_test = dl.attribute(feats1)
                                # print(f'The DL time was {time.time() - start_dl}')
                                # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                start_fa = time.time()
                                fa_attr_test = fa.attribute(feats1)
                                print(f'The FA time was {time.time() - start_fa}')
                                # print('IG + SmoothGrad Attributions:', attributions)
                                # print('Convergence Delta:', delta)

                                # Print
                                for single_feature in range(blud0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features.append(mvp_feature)
                                    cont_features.append(blud0[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_attr_test.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_attr_test[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ig_mvp_features.append(mvp_feature)
                                    ig_cont_features.append(ig_attr_test[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_nt_attr_test.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_nt_attr_test[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ignt_mvp_features.append(mvp_feature)
                                    ignt_cont_features.append(ig_nt_attr_test[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features.append(mvp_feature)
                                #     dl_cont_features.append(dl_attr_test[single_feature, :].cpu().tolist())

                                for single_feature in range(fa_attr_test.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(fa_attr_test[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    fa_mvp_features.append(mvp_feature)
                                    fa_cont_features.append(fa_attr_test[single_feature, :].cpu().tolist())

                                occlusion_count += 1
                            elif not bloods_flag and images_flag:
                                if occlusion_count == 0:
                                    oc = Occlusion(model_wrapper)
                                    sl = Saliency(model_wrapper)
                                    start_oc = time.time()
                                    # Images and features occlusion combined
                                    oc_attributions0 = oc.attribute(image1,
                                                                    sliding_window_shapes=(3, x_shape, x_shape),
                                                                    strides=(3, x_stride, x_stride), target=None,
                                                                    baselines=baseline)
                                    print(f'The OC time was {time.time() - start_oc}')
                                    # Save images
                                    image_grid1 = image1.cpu().numpy()
                                    image_grid1 = reshape_array(image_grid1)
                                    cv2.imwrite(f'{fig_dir}/image_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(image_grid1))

                                    # Save attributions
                                    oc_attributions0 = oc_attributions0.cpu().numpy()
                                    im = reshape_array(oc_attributions0, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                occlusion_count += 1
                            elif bloods_flag and images_flag:
                                print('Running occlusion on Images AND Bloods!')
                                if occlusion_count == 0:
                                    oc = Occlusion(model_wrapper)
                                    sl = Saliency(model_wrapper)
                                    start_oc = time.time()
                                    # Images and features occlusion combined
                                    oc_attributions0, blud0 = oc.attribute((image1, feats1),
                                                                           sliding_window_shapes=((3, x_shape, x_shape), (1,)),
                                                                           strides=((3, x_stride, x_stride), (1,)), target=None,
                                                                           baselines=(baseline, baseline_bloods))
                                    print(f'The OC time was {time.time() - start_oc}')
                                    # Save images
                                    image_grid1 = image1.cpu().numpy()
                                    image_grid1 = reshape_array(image_grid1)
                                    cv2.imwrite(f'{fig_dir}/image_grid1_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid1))

                                    # Save attributions
                                    oc_attributions0 = oc_attributions0.cpu().numpy()
                                    im = reshape_array(oc_attributions0, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                    # Saliency
                                    start_sl = time.time()
                                    sal_attributions0, sal0 = sl.attribute((image1.requires_grad_(),
                                                                            feats1.requires_grad_()), abs=False)
                                    print(f'The Saliency time was {time.time() - start_sl}')
                                # Calculate attribution scores + delta
                                # ig = IntegratedGradients(model)
                                ig = IntegratedGradients(model_wrapper)
                                # dl = DeepLift(model)
                                # gs = GradientShap(model)
                                # fa = FeatureAblation(model_wrapper)
                                ig_nt = NoiseTunnel(ig)
                                # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                start_oc = time.time()
                                _, blud0 = oc.attribute((image1, feats1),
                                                        sliding_window_shapes=((3, image1.shape[2], image1.shape[3]), (1,)),
                                                        strides=((3, image1.shape[2], image1.shape[3]), (1,)),
                                                        target=None,
                                                        baselines=(image1, baseline_bloods))
                                print(f'The OC time was {time.time() - start_oc}')

                                # start_ig = time.time()
                                # _, ig_attr_test = ig.attribute((image1, feats1), n_steps=1)
                                # print(f'The IG time was {time.time() - start_ig}')
                                #
                                # start_ignt = time.time()
                                # _, ig_nt_attr_test = ig_nt.attribute((image1, feats1))
                                # print(f'The IGNT time was {time.time() - start_ignt}')

                                # start_dl = time.time()
                                # _, dl_attr_test, _, _ = dl.attribute((image1, feats1, image2, feats2))
                                # print(f'The DL time was {time.time() - start_dl}')
                                # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                # start_fa = time.time()
                                # _, fa_attr_test = fa.attribute((image1, feats1))
                                # print(f'The FA time was {time.time() - start_fa}')
                                # print('IG + SmoothGrad Attributions:', attributions)
                                # print('Convergence Delta:', delta)

                                # Print
                                for single_feature in range(blud0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features.append(mvp_feature)
                                    cont_features.append(blud0[single_feature, :].cpu().tolist())

                                # for single_feature in range(ig_attr_test.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(ig_attr_test[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     ig_mvp_features.append(mvp_feature)
                                #     ig_cont_features.append(ig_attr_test[single_feature, :].cpu().tolist())
                                #
                                # for single_feature in range(ig_nt_attr_test.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(ig_nt_attr_test[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     ignt_mvp_features.append(mvp_feature)
                                #     ignt_cont_features.append(ig_nt_attr_test[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features.append(mvp_feature)
                                #     dl_cont_features.append(dl_attr_test[single_feature, :].cpu().tolist())

                                # for single_feature in range(fa_attr_test.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(fa_attr_test[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     fa_mvp_features.append(mvp_feature)
                                #     fa_cont_features.append(fa_attr_test[single_feature, :].cpu().tolist())

                                occlusion_count += 1

                    if bloods_flag and multi_flag:
                        running_mvp_features.append(mvp_features)
                        running_cont_features.append(cont_features)
                        running_ig_mvp_features.append(ig_mvp_features)
                        running_ig_cont_features.append(ig_cont_features)
                        running_ignt_mvp_features.append(ignt_mvp_features)
                        running_ignt_cont_features.append(ignt_cont_features)
                        # running_dl_mvp_features.append(dl_mvp_features)
                        # running_dl_cont_features.append(dl_cont_features)
                        running_fa_mvp_features.append(fa_mvp_features)
                        running_fa_cont_features.append(fa_cont_features)

                    elif bloods_flag and not multi_flag:
                        running_mvp_features.append(mvp_features)
                        running_cont_features.append(cont_features)
                        running_ig_mvp_features.append(ig_mvp_features)
                        running_ig_cont_features.append(ig_cont_features)
                        running_ignt_mvp_features.append(ignt_mvp_features)
                        running_ignt_cont_features.append(ignt_cont_features)
                        # running_dl_mvp_features.append(dl_mvp_features)
                        # running_dl_cont_features.append(dl_cont_features)
                        running_fa_mvp_features.append(fa_mvp_features)
                        running_fa_cont_features.append(fa_cont_features)

                        running_mvp_features1.append(mvp_features1)
                        running_cont_features1.append(cont_features1)
                        running_ig_mvp_features1.append(ig_mvp_features1)
                        running_ig_cont_features1.append(ig_cont_features1)
                        running_ignt_mvp_features1.append(ignt_mvp_features1)
                        running_ignt_cont_features1.append(ignt_cont_features1)
                        # running_dl_mvp_features1.append(dl_mvp_features1)
                        # running_dl_cont_features1.append(dl_cont_features1)
                        running_fa_mvp_features1.append(fa_mvp_features1)
                        running_fa_cont_features1.append(fa_cont_features1)

                # Save model
                if SAVE and append_string == 'best' and best_counter < 5:
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
                                'running_val_preds': running_val_preds,
                                'running_val_labels': running_val_labels,
                                'running_val_names': running_val_names,
                                'running_val_roc_aucs': running_val_roc_aucs,
                                'running_val_pr_aucs': running_val_pr_aucs,
                                'overall_val_preds': overall_val_preds,
                                'overall_val_labels': overall_val_labels,
                                'overall_val_names': overall_val_names,
                                'overall_val_roc_aucs': overall_val_roc_aucs,
                                'overall_val_pr_aucs': overall_val_pr_aucs,
                                'running_mvp_features': running_mvp_features,
                                'overall_mvp_features': overall_mvp_features,
                                'best_epochs': best_epochs}, MODEL_PATH)

                elif best_counter >= 5:
                    # Set overalls to best epoch
                    best_epoch = int(np.argmax(running_val_roc_aucs))
                    best_epochs.append(best_epoch)
                    print(f'The best epoch is Epoch {best_epoch}')
                    print(f'The best epoch is {best_epoch} and fold is {fold}')
                    overall_val_roc_aucs.append(running_val_roc_aucs[best_epoch])
                    overall_val_pr_aucs.append(running_val_pr_aucs[best_epoch])
                    overall_val_names.extend(running_val_names[best_epoch])
                    overall_val_preds.extend(running_val_preds[best_epoch])
                    overall_val_labels.extend(running_val_labels[best_epoch])
                    if multi_flag and bloods_flag and occlusion:
                        overall_mvp_features.extend(running_mvp_features[-1])
                        overall_cont_features.extend(running_cont_features[-1])
                        overall_ig_mvp_features.extend(running_ig_mvp_features[-1])
                        overall_ig_cont_features.extend(running_ig_cont_features[-1])
                        overall_ignt_mvp_features.extend(running_ignt_mvp_features[-1])
                        overall_ignt_cont_features.extend(running_ignt_cont_features[-1])
                        # overall_dl_mvp_features.extend(running_dl_mvp_features[-1])
                        # overall_dl_cont_features.extend(running_dl_cont_features[-1])
                        overall_fa_mvp_features.extend(running_fa_mvp_features[-1])
                        overall_fa_cont_features.extend(running_fa_cont_features[-1])

                        overall_mvp_features1.extend(running_mvp_features1[-1])
                        overall_cont_features1.extend(running_cont_features1[-1])
                        overall_ig_mvp_features1.extend(running_ig_mvp_features1[-1])
                        overall_ig_cont_features1.extend(running_ig_cont_features1[-1])
                        overall_ignt_mvp_features1.extend(running_ignt_mvp_features1[-1])
                        overall_ignt_cont_features1.extend(running_ignt_cont_features1[-1])
                        # overall_dl_mvp_features1.extend(running_dl_mvp_features1[-1])
                        # overall_dl_cont_features1.extend(running_dl_cont_features1[-1])
                        overall_fa_mvp_features1.extend(running_fa_mvp_features1[-1])
                        overall_fa_cont_features1.extend(running_fa_cont_features1[-1])

                    elif not multi_flag and bloods_flag and occlusion:
                        print(occlusion)
                        overall_mvp_features.extend(running_mvp_features[-1])
                        overall_cont_features.extend(running_cont_features[-1])
                        overall_ig_mvp_features.extend(running_ig_mvp_features[-1])
                        overall_ig_cont_features.extend(running_ig_cont_features[-1])
                        overall_ignt_mvp_features.extend(running_ignt_mvp_features[-1])
                        overall_ignt_cont_features.extend(running_ignt_cont_features[-1])
                        # overall_dl_mvp_features.extend(running_dl_mvp_features[-1])
                        # overall_dl_cont_features.extend(running_dl_cont_features[-1])
                        overall_fa_mvp_features.extend(running_fa_mvp_features[-1])
                        overall_fa_cont_features.extend(running_fa_cont_features[-1])

                    # Re-save
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                # 'scheduler_state_dict': scheduler.state_dict(),
                                'epoch': epoch,
                                'loss': loss,
                                'running_iter': running_iter,
                                'batch_size': bs,
                                'running_val_preds': running_val_preds,
                                'running_val_labels': running_val_labels,
                                'running_val_names': running_val_names,
                                'running_val_roc_aucs': running_val_roc_aucs,
                                'running_val_pr_aucs': running_val_pr_aucs,
                                'overall_val_preds': overall_val_preds,
                                'overall_val_labels': overall_val_labels,
                                'overall_val_names': overall_val_names,
                                'overall_val_roc_aucs': overall_val_roc_aucs,
                                'overall_val_pr_aucs': overall_val_pr_aucs,
                                'running_mvp_features': running_mvp_features,
                                'overall_mvp_features': overall_mvp_features,
                                'best_epochs': best_epochs}, MODEL_PATH)
                    break

        # Now that this fold's training has ended, want starting points of next fold to reset
        latest_epoch = -1
        latest_fold = 0
        running_iter = 0

        # Print various fold outputs: Sanity check
        # print(f'Fold {fold} val_preds: {val_preds}')
        # print(f'Fold {fold} val_labels: {val_labels}')
        # print(f'Fold {fold} overall_val_roc_aucs: {overall_val_roc_aucs}')
        # print(f'Fold {fold} overall_val_pr_aucs: {overall_val_pr_aucs}')


    ## Totals
    overall_val_labels = np.array(overall_val_labels)
    overall_val_preds = np.array(overall_val_preds)

    overall_val_roc_aucs = np.array(overall_val_roc_aucs)
    overall_val_pr_aucs = np.array(overall_val_pr_aucs)

    # Folds analysis
    print('Labels', len(overall_val_labels), 'Preds', len(overall_val_preds), 'AUCs', len(overall_val_roc_aucs))
    correct = ((overall_val_preds > 0.5).astype(int) == overall_val_labels).sum()
    acc = correct / len(overall_val_labels)

    # Folds AUCs
    print(f'The overall preds is {overall_val_preds.shape}')
    folds_roc_auc = roc_auc_score(overall_val_labels, overall_val_preds)
    precision_folds, recall_folds, _ = precision_recall_curve(overall_val_labels.ravel(), overall_val_preds.ravel())
    folds_pr_auc = auc(recall_folds, precision_folds)
    # print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), folds_roc_auc))
    print('ROC AUC mean:', np.mean(overall_val_roc_aucs), 'std:', np.std(overall_val_roc_aucs))
    print('PR AUC mean:', np.mean(overall_val_pr_aucs), 'std:', np.std(overall_val_pr_aucs))


    all_accs = []
    fold_length = len(overall_val_labels) // FOLDS
    rounded_overall_val_preds = [round(x) for x in np.squeeze(overall_val_preds)]
    for acc in range(FOLDS):
        all_accs.append(accuracy_score(overall_val_labels[acc*fold_length:(acc+1)*fold_length], rounded_overall_val_preds[acc*fold_length:(acc+1)*fold_length]))
    print('Balanced Accuracy:', np.mean(all_accs), 'std:', np.std(all_accs))

    # Store variables
    mean_val_ROC_final, std_val_ROC_final = np.mean(overall_val_roc_aucs), np.std(overall_val_roc_aucs)
    mean_val_PR_final, std_val_PR_final = np.mean(overall_val_pr_aucs), np.std(overall_val_pr_aucs)
    mean_val_BA_final, std_val_BA_final = np.mean(all_accs), np.std(all_accs)

    # Important results!
    print('Val ROC AUC mean:', mean_val_ROC_final, 'std:', std_val_ROC_final)
    print('Val PR AUC mean:', mean_val_PR_final, 'std:', std_val_PR_final)
    print('Val Balanced Accuracy:', mean_val_BA_final, 'std:', std_val_BA_final)

    if multi_flag and bloods_flag:
        print(
            f'Length of overall_val_names, overall_val_labels, overall_val_preds, overall_mvp_features are {len(overall_val_names)},'
            f'{len(overall_val_labels)}, {len(overall_val_preds)}, {len(overall_mvp_features)},'
            f'{len(overall_fa_mvp_features)}, {len(overall_fa_cont_features)}',
            f'{len(overall_ignt_mvp_features)}, {len(overall_ignt_cont_features)}',
            f'{len(overall_ig_mvp_features)}, {len(overall_ig_cont_features)}',
            # f'{len(overall_dl_mvp_features)}, {len(overall_dl_cont_features)}, '
            f'{len(overall_mvp_features1)},'
            f'{len(overall_fa_mvp_features1)}, {len(overall_fa_cont_features1)}',
            f'{len(overall_ignt_mvp_features1)}, {len(overall_ignt_cont_features1)}',
            f'{len(overall_ig_mvp_features1)}, {len(overall_ig_cont_features1)}'
            # f'{len(overall_dl_mvp_features1)}, {len(overall_dl_cont_features1)}'
        )

        sub = pd.DataFrame(
            pad_dict_list({"Filename": overall_val_names, "Died": overall_val_labels.tolist(), "Pred": overall_val_preds.tolist(),
             "MVP_feat": overall_mvp_features, 'Cont_feat': overall_cont_features,
             "IG_MVP_feat": overall_ig_mvp_features, 'IG_Cont_feat': overall_ig_cont_features,
             "IGNT_MVP_feat": overall_ignt_mvp_features, 'IGNT_Cont_feat': overall_ignt_cont_features,
             # "DL_MVP_feat": overall_dl_mvp_features, 'DL_Cont_feat': overall_dl_cont_features,
             "FA_MVP_feat": overall_fa_mvp_features, 'FA_Cont_feat': overall_fa_cont_features,
             "MVP_feat1": overall_mvp_features1, 'Cont_feat1': overall_cont_features1,
             "IG_MVP_feat1": overall_ig_mvp_features1, 'IG_Cont_feat1': overall_ig_cont_features1,
             "IGNT_MVP_feat1": overall_ignt_mvp_features1, 'IGNT_Cont_feat1': overall_ignt_cont_features1,
             # "DL_MVP_feat1": overall_dl_mvp_features1, 'DL_Cont_feat1': overall_dl_cont_features1,
             "FA_MVP_feat1": overall_fa_mvp_features1, 'FA_Cont_feat1': overall_fa_cont_features1
             }))
    elif not multi_flag and bloods_flag:
        print(
            f'Length of overall_val_names, overall_val_labels, overall_val_preds, overall_mvp_features are {len(overall_val_names)},'
            f'{len(overall_val_labels)}, {len(overall_val_preds)}, {len(overall_mvp_features)},'
            f'{len(overall_fa_mvp_features)}, {len(overall_fa_cont_features)}',
            f'{len(overall_ignt_mvp_features)}, {len(overall_ignt_cont_features)}',
            f'{len(overall_ig_mvp_features)}, {len(overall_ig_cont_features)}',
            # f'{len(overall_dl_mvp_features)}, {len(overall_dl_cont_features)}'
        )
        sub = pd.DataFrame(pad_dict_list({"Filename": overall_val_names, "Died": overall_val_labels.tolist(), "Pred": overall_val_preds.tolist(),
                            "MVP_feat": overall_mvp_features, 'Cont_feat': overall_cont_features,
                            "IG_MVP_feat": overall_ig_mvp_features, 'IG_Cont_feat': overall_ig_cont_features,
                            "IGNT_MVP_feat": overall_ignt_mvp_features, 'IGNT_Cont_feat': overall_ignt_cont_features,
                            # "DL_MVP_feat": overall_dl_mvp_features, 'DL_Cont_feat': overall_dl_cont_features,
                            "FA_MVP_feat": overall_fa_mvp_features, 'FA_Cont_feat': overall_fa_cont_features
                            }))
    else:
        sub = pd.DataFrame({"Filename": overall_val_names, "Died": overall_val_labels.tolist(), "Pred": overall_val_preds.tolist()
                            })
    sub[str(temp_bloods.columns.to_list())] = 0.0
    sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

    ## Plot
    # Compute ROC curve and ROC area for each class
    class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(overall_val_labels.ravel(), overall_val_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute PR curve and PR area for each class
    precision_tot = dict()
    recall_tot = dict()
    pr_auc = dict()

    # Compute micro-average precision-recall curve and PR area
    precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(overall_val_labels.ravel(), overall_val_preds.ravel())
    pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])
    no_skill = len(overall_val_labels[overall_val_labels == 1]) / len(overall_val_labels)

    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red']

    # Plot ROC-AUC for different classes:
    plt.figure()
    plt.axis('square')
    for classID, key in enumerate(fpr.keys()):
        lw = 2
        plt.plot(fpr[key], tpr[key], color=colors[classID],  # 'darkorange',
                 lw=lw, label=f'Overall ROC curve (area = {roc_auc[key]: .2f})')
        plt.title(f'Overall ROC-AUC', fontsize=18)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_PATH, f'roc-bs{bs}-logreg.png'), dpi=300)

    # PR plot
    plt.figure()
    plt.axis('square')
    for classID, key in enumerate(precision_tot.keys()):
        lw = 2
        plt.plot(recall_tot[key], precision_tot[key], color=colors[classID],  # color='darkblue',
                 lw=lw, label=f'Overall PR curve (area = {pr_auc[key]: .2f})')
        plt.title(f'Overall PR-AUC', fontsize=18)
        # plt.plot([0, 1], [0, 0], lw=lw, linestyle='--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.legend(loc="lower right")
    fig_name = f'precision-recall-bs{bs}-logreg.png'
    plt.savefig(os.path.join(SAVE_PATH, fig_name), dpi=300)

    # Save relevant data to csv
    # overall_val_labels = [x[0] for x in overall_val_labels]
    # overall_val_preds = [x[0] for x in overall_val_preds]
    # sub = pd.DataFrame({"Filename": overall_val_names, "Died": overall_val_labels, "Pred": overall_val_preds, "MVP_feat": overall_mvp_features})
    # sub_name = f'preds-bs{bs}-logreg-{arguments.job_name}.csv'
    # sub.to_csv(os.path.join(SAVE_PATH, sub_name), index=False)

## Test
do_test = True
if do_test:
    # Instantiate model
    singlemodel = SingleModel(encoder).cuda()
    model = CombinedModel(singlemodel).cuda()
    model = nn.DataParallel(model)
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    latest_model_file = max(model_files, key=os.path.getctime)
    latest_checkpoint = torch.load(os.path.join(SAVE_PATH, latest_model_file), map_location=torch.device('cuda:0'))
    best_epochs = latest_checkpoint['best_epochs']
    print('Best epochs', best_epochs)
    # Load best models
    if not do_train:
        print('Testing directly!')

    test_df, _ = prepare_data(test_df.reset_index(), bloods_cols) #, # myscaler=train_scaler)
    print('Test data:', test_df.shape)
    test_dataset = ImageDataset(test_df, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=int(bs / 4), num_workers=8, shuffle=False)
    from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

    y_pred = 0
    test_accs = []
    test_aucs = []
    for fold in range(FOLDS):
        print('\nFOLD', fold)
        test_running_val_preds = []
        test_running_val_labels = []
        test_running_val_names = []
        test_running_val_roc_aucs = []
        test_running_val_pr_aucs = []
        test_running_mvp_features = []

        test_running_cont_features = []
        test_running_ig_mvp_features = []
        test_running_ig_cont_features = []
        test_running_ignt_mvp_features = []
        test_running_ignt_cont_features = []
        # test_running_dl_mvp_features = []
        # test_running_dl_cont_features = []
        test_running_fa_mvp_features = []
        test_running_fa_cont_features = []

        test_running_cont_features1 = []
        test_running_ig_mvp_features1 = []
        test_running_ig_cont_features1 = []
        test_running_ignt_mvp_features1 = []
        test_running_ignt_cont_features1 = []
        # test_running_dl_mvp_features1 = []
        # test_running_dl_cont_features1 = []
        test_running_fa_mvp_features1 = []
        test_running_fa_cont_features1 = []

        ## Load best model!
        # Just need best epochs to proceed (In theory!)
        MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{best_epochs[fold]}_fold_{fold}.pth')
        optimizer = RangerLars(model.parameters())
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda:0'))
            # Adjust key names
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # keys_list = checkpoint['model_state_dict'].keys()
            # new_dict = checkpoint['model_state_dict'].copy()
            # for name in keys_list:
            #     new_dict[name[7:]] = checkpoint['model_state_dict'][name]
            #     del new_dict[name]
            # model.load_state_dict(checkpoint['model_state_dict'])
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded:', MODEL_PATH)

        res_name, res_prob, res_label = [], [], []
        model.cuda()
        model.eval()
        test_counter = 0
        with torch.no_grad():
            if multi_flag:
                for pid in test_df['patient_pseudo_id'].unique():
                    #print(pid)
                    pid_df = test_df[test_df['patient_pseudo_id'] == pid].reset_index(drop=True)
                    pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                    pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                    n_images = pid_df.shape[0]
                    outs = []
                    labs = []
                    for n in range(n_images):
                        ind1 = 0
                        ind2 = n
                        time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                        time2 = pid_df.rel_datetime[ind2].astype(np.float32)

                        image1, image2 = np.array([]), np.array([])
                        feats1, feats2 = np.array([]), np.array([])
                        if bloods_flag:
                            # Features
                            feats1 = get_feats(pid_df, ind1, aug=False)
                            feats2 = get_feats(pid_df, ind2, aug=False)
                            feats1 = np.append(feats1, time1)
                            feats2 = np.append(feats2, time2)
                        else:
                            feats1, feats2 = np.array([]), np.array([])
                        if images_flag:
                            # Image
                            image1 = get_image(pid_df, ind1, tta_transform)
                            image2 = get_image(pid_df, ind2, tta_transform)
                        else:
                            # image1, image2 = torch.FloatTensor(), torch.FloatTensor()
                            image1, image2 = torch.FloatTensor(), torch.FloatTensor()

                        # Label
                        labels = pid_df.Died[ind1]
                        labels = torch.Tensor([labels]).cuda()
                        labels = labels.unsqueeze(1).float()

                        # Features
                        feats1 = torch.Tensor(feats1).cuda()
                        feats1 = feats1.unsqueeze(0)
                        feats2 = torch.Tensor(feats2).cuda()
                        feats2 = feats2.unsqueeze(0)

                        # Images
                        image1, image2 = image1.cuda(), image2.cuda()
                        # image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)
                        out = model(image1, feats1, image2, feats2)
                        out = torch.sigmoid(out)
                        outs += out.cpu().numpy().tolist()
                        labs += labels.cpu().numpy().tolist()
                    outs_mean = np.mean(outs)
                    test_running_val_preds += outs
                    test_running_val_labels += labs
                    test_running_val_names += [pid] * n_images
                    # print(f'test running stuff: {len(test_running_val_preds)}, {len(test_running_val_labels)},'
                    #       f'{len(test_running_val_names)}')
                    test_counter += 1
            else:
                for i, sample in enumerate(test_loader):
                    early_images, names, labels, early_bloods = sample[0], sample[1], sample[2], sample[3]
                    # print(f'Test images shape is {early_images.shape}')
                    later_images, later_bloods = None, None

                    labels = labels.cuda()
                    labels = labels.unsqueeze(1).float()
                    early_bloods = early_bloods.cuda()
                    early_bloods = early_bloods.float()

                    # TTA
                    if len(early_images.size()) == 5:
                        batch_size, n_crops, c, h, w = early_images.size()
                        early_images = early_images.view(-1, c, h, w)
                        if bloods_flag:
                            _, n_feats = early_bloods.size()
                            early_bloods = early_bloods.repeat(1, n_crops).view(-1, n_feats)
                        out = model(early_images, early_bloods, later_images, later_bloods)
                        out = out.view(batch_size, n_crops, -1).mean(1)
                    else:
                        out = model(early_images, early_bloods, later_images, later_bloods)
                    out = torch.sigmoid(out)

                    # Save validation output for post all folds training aggregation
                    test_running_val_preds += out.cpu().numpy().tolist()
                    test_running_val_labels += labels.cpu().numpy().tolist()
                    test_running_val_names += names

                    acc = ((out > 0.5).int() == labels).sum().item()
                    # correct += ((out > 0.5).int() == labels).sum().item()
                    test_counter += 1

            test_running_val_labels = np.array(test_running_val_labels)
            test_running_val_preds = np.array(test_running_val_preds)
            test_auc = roc_auc_score(test_running_val_labels, test_running_val_preds)
            test_acc = accuracy_score(test_running_val_labels, (test_running_val_preds > 0.5).astype(int))
            test_accs.append(test_acc)
            test_aucs.append(test_auc)
            print('Accuracy:', test_acc, 'AUC:', test_auc)
            y_pred += test_running_val_preds
        if occ_flag:
            mvp_features = []
            ig_mvp_features = []
            ignt_mvp_features = []
            # dl_mvp_features = []
            fa_mvp_features = []
            cont_features = []
            ig_cont_features = []
            ignt_cont_features = []
            # dl_cont_features = []
            fa_cont_features = []

            mvp_features1 = []
            ig_mvp_features1 = []
            ignt_mvp_features1 = []
            # dl_mvp_features1 = []
            fa_mvp_features1 = []
            cont_features1 = []
            ig_cont_features1 = []
            ignt_cont_features1 = []
            # dl_cont_features1 = []
            fa_cont_features1 = []
            with torch.no_grad():
                occlusion_count = 0
                if multi_flag:
                    print('Multi validation')
                    for pid in test_df['patient_pseudo_id'].unique():
                        # print(pid)
                        pid_df = test_df[test_df['patient_pseudo_id'] == pid].reset_index(drop=True)
                        pid_df['CXR_datetime'] = pd.to_datetime(pid_df.CXR_datetime, dayfirst=True)
                        pid_df = pid_df.sort_values(by=['CXR_datetime'], ascending=True).reset_index(drop=True)
                        n_images = pid_df.shape[0]
                        # print(n_images, 'images')
                        for n in range(n_images):
                            ind1 = 0
                            ind2 = n
                            time1 = pid_df.rel_datetime[ind1].astype(np.float32)
                            time2 = pid_df.rel_datetime[ind2].astype(np.float32)

                            image1, image2 = np.array([]), np.array([])
                            feats1, feats2 = np.array([]), np.array([])
                            if bloods_flag:
                                # Features
                                feats1 = get_feats(pid_df, ind1, aug=False)
                                feats2 = get_feats(pid_df, ind2, aug=False)
                                feats1 = np.append(feats1, time1)
                                feats2 = np.append(feats2, time2)
                            else:
                                feats1, feats2 = np.array([]), np.array([])
                            if images_flag:
                                # Image
                                image1 = get_image(pid_df, ind1, tta_transform)
                                image2 = get_image(pid_df, ind2, tta_transform)
                            else:
                                image1, image2 = torch.FloatTensor(), torch.FloatTensor()

                            # Label
                            labels = pid_df.Died[ind1]
                            labels = torch.Tensor([labels]).cuda()
                            labels = labels.unsqueeze(1).float()

                            # Features
                            feats1 = torch.Tensor(feats1).cuda()
                            feats1 = feats1.unsqueeze(0)
                            feats2 = torch.Tensor(feats2).cuda()
                            feats2 = feats2.unsqueeze(0)

                            # Images
                            image1, image2 = image1.cuda(), image2.cuda()
                            image1, image2 = image1.unsqueeze(0), image2.unsqueeze(0)

                            # Always create
                            baseline_bloods = torch.zeros_like(feats1).cuda().float()
                            baseline = torch.zeros_like(image1).cuda()

                            # Int for bloods only
                            if bloods_flag and not images_flag:
                                # Calculate attribution scores + delta
                                oc = Occlusion(model_wrapper)
                                # ig = IntegratedGradients(model)
                                ig = IntegratedGradients(model_wrapper)
                                # dl = DeepLift(model)
                                # gs = GradientShap(model)
                                fa = FeatureAblation(model_wrapper)
                                ig_nt = NoiseTunnel(ig)
                                # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                start_oc = time.time()
                                blud0, blud1 = oc.attribute((feats1, feats2), sliding_window_shapes=((1,), (1,)),
                                                            strides=((1,), (1,)), target=None,
                                                            baselines=(baseline_bloods, baseline_bloods))
                                print(f'The OC time was {time.time() - start_oc}')

                                start_ig = time.time()
                                ig_attr_test0, ig_attr_test1 = ig.attribute((feats1, feats2), n_steps=50)
                                print(f'The IG time was {time.time() - start_ig}')

                                start_ignt = time.time()
                                ig_nt_attr_test0, ig_nt_attr_test1 = ig_nt.attribute((feats1, feats2))
                                print(f'The IGNT time was {time.time() - start_ignt}')

                                # start_dl = time.time()
                                # _, dl_attr_test0, _, dl_attr_test1 = dl.attribute((image1, feats1, image2, feats2))
                                # print(f'The DL time was {time.time() - start_dl}')
                                # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                # start_fa = time.time()
                                # fa_attr_test0, fa_attr_test1 = fa.attribute((feats1, feats2))
                                # print(f'The FA time was {time.time() - start_fa}')
                                # print('IG + SmoothGrad Attributions:', attributions)
                                # print('Convergence Delta:', delta)

                                # Print
                                for single_feature in range(blud0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features.append(mvp_feature)
                                    cont_features.append(blud0[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_attr_test0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_attr_test0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ig_mvp_features.append(mvp_feature)
                                    ig_cont_features.append(ig_attr_test0[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_nt_attr_test0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_nt_attr_test0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ignt_mvp_features.append(mvp_feature)
                                    ignt_cont_features.append(ig_nt_attr_test0[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test0.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test0[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features.append(mvp_feature)
                                #     dl_cont_features.append(dl_attr_test0[single_feature, :].cpu().tolist())

                                # for single_feature in range(fa_attr_test0.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(fa_attr_test0[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     fa_mvp_features.append(mvp_feature)
                                #     fa_cont_features.append(fa_attr_test0[single_feature, :].cpu().tolist())

                                # Again, for the rest
                                for single_feature in range(blud1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features1.append(mvp_feature)
                                    cont_features1.append(blud1[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_attr_test1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_attr_test1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ig_mvp_features1.append(mvp_feature)
                                    ig_cont_features1.append(ig_attr_test1[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_nt_attr_test1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_nt_attr_test1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ignt_mvp_features1.append(mvp_feature)
                                    ignt_cont_features1.append(ig_nt_attr_test1[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test1.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test1[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features1.append(mvp_feature)
                                #     dl_cont_features1.append(dl_attr_test1[single_feature, :].cpu().tolist())

                                # for single_feature in range(fa_attr_test1.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(fa_attr_test1[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     fa_mvp_features1.append(mvp_feature)
                                #     fa_cont_features1.append(fa_attr_test1[single_feature, :].cpu().tolist())
                            elif not bloods_flag and images_flag:
                                if occlusion_count == 0:
                                    oc = Occlusion(model_wrapper)
                                    sl = Saliency(model_wrapper)
                                    start_oc = time.time()
                                    # Images and features occlusion combined
                                    oc_attributions0, oc_attributions1 = oc.attribute((image1, image2),
                                                                                      sliding_window_shapes=(
                                                                                          (3, x_shape, x_shape),
                                                                                          (3, x_shape, x_shape)),
                                                                                      strides=((3, x_stride, x_stride),
                                                                                               (3, x_stride, x_stride)),
                                                                                      target=None,
                                                                                      baselines=(baseline, baseline))
                                    print(f'The OC time was {time.time() - start_oc}')

                                    # Save images
                                    image_grid1 = image1.cpu().numpy()
                                    image_grid1 = reshape_array(image_grid1)
                                    cv2.imwrite(f'{fig_dir}/image_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid1))

                                    image_grid2 = image2.cpu().numpy()
                                    image_grid2 = reshape_array(image_grid2)
                                    cv2.imwrite(f'{fig_dir}/image_grid2_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid2))

                                    # Save attributions
                                    oc_attributions0 = oc_attributions0.cpu().numpy()
                                    # print(f'The attributions shape is {oc_attributions0.shape}')
                                    im = reshape_array(oc_attributions0, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                    oc_attributions1 = oc_attributions1.cpu().numpy()
                                    im = reshape_array(oc_attributions1, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.int8(im))

                                    # Saliency
                                    start_sl = time.time()
                                    sal_attributions0, sal_attributions1 = sl.attribute(
                                        (image1.requires_grad_(), image2.requires_grad_()), abs=False)
                                    print(f'The Saliency time was {time.time() - start_sl}')
                            elif bloods_flag and images_flag:
                                print(occlusion_count)
                                if occlusion_count == 0:
                                    oc = Occlusion(model)
                                    sl = Saliency(model)
                                    start_oc = time.time()
                                    # Images and features occlusion combined
                                    oc_attributions0, blud0, oc_attributions1, blud1 = oc.attribute(
                                        (image1.requires_grad_(), feats1, image2.requires_grad_(), feats2),
                                        sliding_window_shapes=(
                                            (3, x_shape, x_shape), (1,),
                                            (3, x_shape, x_shape), (1,)),
                                        strides=((3, x_stride, x_stride), (1,),
                                                 (3, x_stride, x_stride), (1,)), target=None,
                                        baselines=(baseline, baseline_bloods, baseline, baseline_bloods))
                                    print(f'The OC time was {time.time() - start_oc}')

                                    # Save images
                                    image_grid1 = image1.cpu().numpy()
                                    image_grid1 = reshape_array(image_grid1)
                                    cv2.imwrite(f'{fig_dir}/image_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid1))

                                    image_grid2 = image2.cpu().numpy()
                                    image_grid2 = reshape_array(image_grid2)
                                    cv2.imwrite(f'{fig_dir}/image_grid2_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.uint8(image_grid2))

                                    # Save attributions
                                    oc_attributions0 = oc_attributions0.cpu().numpy()
                                    im = reshape_array(oc_attributions0, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                    oc_attributions1 = oc_attributions1.cpu().numpy()
                                    im = reshape_array(oc_attributions1, colormap_grid=True)
                                    cv2.imwrite(f'{fig_dir}/oc_attributions_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png', np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                    # Saliency
                                    start_sl = time.time()
                                    sal_attributions0, _, sal_attributions1, _ = sl.attribute((image1.requires_grad_(),
                                                                                               feats1,
                                                                                               image2.requires_grad_(),
                                                                                               feats2), abs=False)
                                    print(f'The Saliency time was {time.time() - start_sl}')
                                # Calculate attribution scores + delta
                                # ig = IntegratedGradients(model)
                                ig = IntegratedGradients(model)
                                # dl = DeepLift(model)
                                # gs = GradientShap(model)
                                fa = FeatureAblation(model)
                                ig_nt = NoiseTunnel(ig)
                                # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                                start_oc = time.time()
                                _, blud0, _, blud1 = oc.attribute((image1, feats1, image2, feats2),
                                                                  sliding_window_shapes=(
                                                                  (3, image1.shape[2], image1.shape[3]), (1,),
                                                                  (3, image1.shape[2], image1.shape[3]), (1,)),
                                                                  strides=((3, image1.shape[2], image1.shape[3]), (1,),
                                                                           (3, image1.shape[2], image1.shape[3]), (1,)),
                                                                  target=None,
                                                                  baselines=(
                                                                      image1, baseline_bloods, image2, baseline_bloods))
                                print(f'The OC time was {time.time() - start_oc}')

                                start_ig = time.time()
                                _, ig_attr_test0, _, ig_attr_test1 = ig.attribute((image1, feats1, image2, feats2),
                                                                                  n_steps=50)
                                print(f'The IG time was {time.time() - start_ig}')

                                start_ignt = time.time()
                                _, ig_nt_attr_test0, _, ig_nt_attr_test1 = ig_nt.attribute((feats1, feats2))
                                print(f'The IGNT time was {time.time() - start_ignt}')

                                # start_dl = time.time()
                                # _, dl_attr_test0, _, dl_attr_test1 = dl.attribute((image1, feats1, image2, feats2))
                                # print(f'The DL time was {time.time() - start_dl}')
                                # # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                                start_fa = time.time()
                                _, fa_attr_test0, _, fa_attr_test1 = fa.attribute((image1, feats1, image2, feats2))
                                print(f'The FA time was {time.time() - start_fa}')
                                # print('IG + SmoothGrad Attributions:', attributions)
                                # print('Convergence Delta:', delta)

                                # Print
                                for single_feature in range(blud0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features.append(mvp_feature)
                                    cont_features.append(blud0[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_attr_test0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_attr_test0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ig_mvp_features.append(mvp_feature)
                                    ig_cont_features.append(ig_attr_test0[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_nt_attr_test0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_nt_attr_test0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ignt_mvp_features.append(mvp_feature)
                                    ignt_cont_features.append(ig_nt_attr_test0[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test0.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test0[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features.append(mvp_feature)
                                #     dl_cont_features.append(dl_attr_test0[single_feature, :].cpu().tolist())

                                for single_feature in range(fa_attr_test0.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(fa_attr_test0[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    fa_mvp_features.append(mvp_feature)
                                    fa_cont_features.append(fa_attr_test0[single_feature, :].cpu().tolist())

                                # Again, for the rest
                                for single_feature in range(blud1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(blud1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    mvp_features1.append(mvp_feature)
                                    cont_features1.append(blud1[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_attr_test1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_attr_test1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ig_mvp_features1.append(mvp_feature)
                                    ig_cont_features1.append(ig_attr_test1[single_feature, :].cpu().tolist())

                                for single_feature in range(ig_nt_attr_test1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(ig_nt_attr_test1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    ignt_mvp_features1.append(mvp_feature)
                                    ignt_cont_features1.append(ig_nt_attr_test1[single_feature, :].cpu().tolist())

                                # for single_feature in range(dl_attr_test1.shape[0]):
                                #     mvp_feature = temp_bloods.columns[
                                #         int(np.argmax(torch.abs(dl_attr_test1[single_feature, :]).cpu()))]
                                #     # print(f'The most valuable feature was {mvp_feature}')
                                #     dl_mvp_features1.append(mvp_feature)
                                #     dl_cont_features1.append(dl_attr_test1[single_feature, :].cpu().tolist())

                                for single_feature in range(fa_attr_test1.shape[0]):
                                    mvp_feature = temp_bloods.columns[
                                        int(np.argmax(torch.abs(fa_attr_test1[single_feature, :]).cpu()))]
                                    # print(f'The most valuable feature was {mvp_feature}')
                                    fa_mvp_features1.append(mvp_feature)
                                    fa_cont_features1.append(fa_attr_test1[single_feature, :].cpu().tolist())
                else:
                    print('Not Multi!')
                    for i, sample in enumerate(test_loader):
                        print('Int loader iteration!')
                        image1, names, labels, feats1 = sample[0], sample[1], sample[2], sample[3]
                        image2, feats2 = torch.FloatTensor(), torch.FloatTensor()
                        image1 = image1.cuda()[0:4, ...]  # [None, ...]
                        labels = labels.cuda()[0:4, ...]  # [None, ...]
                        labels = labels.unsqueeze(1).float()
                        feats1 = feats1.cuda()[0:4, ...]  # [None, ...]
                        feats1 = feats1.float()

                        # print(f'The image shape is {image1.shape}')

                        # Account for tta: Take first image (non-augmented)
                        # Label does not need to be touched because it is obv. the same for this image regardless of tta
                        # Set a baseline
                        baseline = torch.zeros_like(image1).cuda().float()
                        baseline_bloods = torch.zeros_like(feats1).cuda().float()

                        # Int for bloods only
                        if bloods_flag and not images_flag:
                            print('Using Bloods!')
                            # Calculate attribution scores + delta
                            # ig = IntegratedGradients(model)
                            oc = Occlusion(model_wrapper)
                            ig = IntegratedGradients(model_wrapper)
                            # dl = DeepLift(model_wrapper)
                            # gs = GradientShap(model)
                            fa = FeatureAblation(model_wrapper)
                            ig_nt = NoiseTunnel(ig)
                            # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                            start_oc = time.time()
                            print(image1.shape, image2.shape, feats1.shape, feats2.shape)
                            blud0 = oc.attribute(feats1,
                                                 sliding_window_shapes=(1,),
                                                 strides=(1,), target=None,
                                                 baselines=baseline_bloods)
                            print(f'The OC time was {time.time() - start_oc}')

                            start_ig = time.time()
                            ig_attr_test = ig.attribute(feats1, n_steps=50)
                            print(f'The IG time was {time.time() - start_ig}')

                            start_ignt = time.time()
                            ig_nt_attr_test = ig_nt.attribute(feats1)
                            print(f'The IGNT time was {time.time() - start_ignt}')

                            # start_dl = time.time()
                            # dl_attr_test = dl.attribute(feats1)
                            # print(f'The DL time was {time.time() - start_dl}')
                            # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                            start_fa = time.time()
                            fa_attr_test = fa.attribute(feats1)
                            print(f'The FA time was {time.time() - start_fa}')
                            # print('IG + SmoothGrad Attributions:', attributions)
                            # print('Convergence Delta:', delta)

                            # Print
                            for single_feature in range(blud0.shape[0]):
                                mvp_feature = temp_bloods.columns[
                                    int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                # print(f'The most valuable feature was {mvp_feature}')
                                mvp_features.append(mvp_feature)
                                cont_features.append(blud0[single_feature, :].cpu().tolist())

                            for single_feature in range(ig_attr_test.shape[0]):
                                mvp_feature = temp_bloods.columns[
                                    int(np.argmax(torch.abs(ig_attr_test[single_feature, :]).cpu()))]
                                # print(f'The most valuable feature was {mvp_feature}')
                                ig_mvp_features.append(mvp_feature)
                                ig_cont_features.append(ig_attr_test[single_feature, :].cpu().tolist())

                            for single_feature in range(ig_nt_attr_test.shape[0]):
                                mvp_feature = temp_bloods.columns[
                                    int(np.argmax(torch.abs(ig_nt_attr_test[single_feature, :]).cpu()))]
                                # print(f'The most valuable feature was {mvp_feature}')
                                ignt_mvp_features.append(mvp_feature)
                                ignt_cont_features.append(ig_nt_attr_test[single_feature, :].cpu().tolist())

                            # for single_feature in range(dl_attr_test.shape[0]):
                            #     mvp_feature = temp_bloods.columns[
                            #         int(np.argmax(torch.abs(dl_attr_test[single_feature, :]).cpu()))]
                            #     # print(f'The most valuable feature was {mvp_feature}')
                            #     dl_mvp_features.append(mvp_feature)
                            #     dl_cont_features.append(dl_attr_test[single_feature, :].cpu().tolist())

                            for single_feature in range(fa_attr_test.shape[0]):
                                mvp_feature = temp_bloods.columns[
                                    int(np.argmax(torch.abs(fa_attr_test[single_feature, :]).cpu()))]
                                # print(f'The most valuable feature was {mvp_feature}')
                                fa_mvp_features.append(mvp_feature)
                                fa_cont_features.append(fa_attr_test[single_feature, :].cpu().tolist())

                            occlusion_count += 1
                        elif not bloods_flag and images_flag:
                            print('Using Images!')
                            if occlusion_count == 0:
                                oc = Occlusion(model_wrapper)
                                sl = Saliency(model_wrapper)
                                start_oc = time.time()
                                # Images and features occlusion combined
                                oc_attributions0 = oc.attribute(image1,
                                                                sliding_window_shapes=(3, x_shape, x_shape),
                                                                strides=(3, x_stride, x_stride), target=None,
                                                                baselines=baseline)
                                print(f'The OC time was {time.time() - start_oc}')

                                # Save images
                                image_grid1 = image1.cpu().numpy()
                                image_grid1 = reshape_array(image_grid1)
                                cv2.imwrite(f'{fig_dir}/image_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png',
                                            np.uint8(image_grid1))

                                # Save attributions
                                oc_attributions0 = oc_attributions0.cpu().numpy()
                                print(f'The attributions shape is {oc_attributions0.shape}')
                                print((oc_attributions0[0, 0, ...] == oc_attributions0[0, 1, ...]).sum(), oc_attributions0[0, 0, ...].size)
                                im = reshape_array(oc_attributions0, colormap_grid=True)
                                cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png',
                                            np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))
                                print(f'The attributions shape is {np.np.int8(im).shape}')

                            occlusion_count += 1
                        elif bloods_flag and images_flag:
                            print('Running occlusion on Images AND Bloods!')
                            if occlusion_count == 0:
                                oc = Occlusion(model_wrapper)
                                sl = Saliency(model_wrapper)
                                start_oc = time.time()
                                # Images and features occlusion combined
                                oc_attributions0, blud0 = oc.attribute((image1, feats1),
                                                                       sliding_window_shapes=(
                                                                       (3, x_shape, x_shape), (1,)),
                                                                       strides=((3, x_stride, x_stride), (1,)),
                                                                       target=None,
                                                                       baselines=(baseline, baseline_bloods))
                                print(f'The OC time was {time.time() - start_oc}')
                                
                                # Save images
                                image_grid1 = image1.cpu().numpy()
                                image_grid1 = reshape_array(image_grid1)
                                cv2.imwrite(f'{fig_dir}/image_grid1_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png',
                                            np.uint8(image_grid1))

                                # Save attributions
                                oc_attributions0 = oc_attributions0.cpu().numpy()
                                im = reshape_array(oc_attributions0, colormap_grid=True)
                                cv2.imwrite(f'{fig_dir}/oc_attributions_grid0_test_Labels_{labels.cpu().int().tolist()}_fold_{fold}.png',
                                            np.np.int8(im)); np.save(f'{fig_dir}/oc_attributions_array0_Labels_{labels.cpu().int().tolist()}_fold_{fold}.npy', np.np.int8(im))

                                # Saliency
                                start_sl = time.time()
                                sal_attributions0, sal0 = sl.attribute((image1.requires_grad_(),
                                                                        feats1.requires_grad_()), abs=False)
                                print(f'The Saliency time was {time.time() - start_sl}')
                            # Calculate attribution scores + delta
                            # ig = IntegratedGradients(model)
                            ig = IntegratedGradients(model_wrapper)
                            # dl = DeepLift(model)
                            # gs = GradientShap(model)
                            # fa = FeatureAblation(model_wrapper)
                            ig_nt = NoiseTunnel(ig)
                            # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)

                            start_oc = time.time()
                            _, blud0 = oc.attribute((image1, feats1),
                                                    sliding_window_shapes=((3, image1.shape[2], image1.shape[3]), (1,)),
                                                    strides=((3, image1.shape[2], image1.shape[3]), (1,)),
                                                    target=None,
                                                    baselines=(image1, baseline_bloods))
                            print(f'The OC time was {time.time() - start_oc}')

                            # start_ig = time.time()
                            # _, ig_attr_test = ig.attribute((image1, feats1), n_steps=1)
                            # print(f'The IG time was {time.time() - start_ig}')
                            #
                            # start_ignt = time.time()
                            # _, ig_nt_attr_test = ig_nt.attribute((image1, feats1))
                            # print(f'The IGNT time was {time.time() - start_ignt}')

                            # start_dl = time.time()
                            # _, dl_attr_test, _, _ = dl.attribute((image1, feats1, image2, feats2))
                            # print(f'The DL time was {time.time() - start_dl}')
                            # gs_attr_test = gs.attribute((feats1, feats2), X_train)

                            # start_fa = time.time()
                            # _, fa_attr_test = fa.attribute((image1, feats1))
                            # print(f'The FA time was {time.time() - start_fa}')
                            # print('IG + SmoothGrad Attributions:', attributions)
                            # print('Convergence Delta:', delta)

                            # Print
                            for single_feature in range(blud0.shape[0]):
                                mvp_feature = temp_bloods.columns[
                                    int(np.argmax(torch.abs(blud0[single_feature, :]).cpu()))]
                                # print(f'The most valuable feature was {mvp_feature}')
                                mvp_features.append(mvp_feature)
                                cont_features.append(blud0[single_feature, :].cpu().tolist())

                            # for single_feature in range(ig_attr_test.shape[0]):
                            #     mvp_feature = temp_bloods.columns[
                            #         int(np.argmax(torch.abs(ig_attr_test[single_feature, :]).cpu()))]
                            #     # print(f'The most valuable feature was {mvp_feature}')
                            #     ig_mvp_features.append(mvp_feature)
                            #     ig_cont_features.append(ig_attr_test[single_feature, :].cpu().tolist())
                            #
                            # for single_feature in range(ig_nt_attr_test.shape[0]):
                            #     mvp_feature = temp_bloods.columns[
                            #         int(np.argmax(torch.abs(ig_nt_attr_test[single_feature, :]).cpu()))]
                            #     # print(f'The most valuable feature was {mvp_feature}')
                            #     ignt_mvp_features.append(mvp_feature)
                            #     ignt_cont_features.append(ig_nt_attr_test[single_feature, :].cpu().tolist())

                            # for single_feature in range(dl_attr_test.shape[0]):
                            #     mvp_feature = temp_bloods.columns[
                            #         int(np.argmax(torch.abs(dl_attr_test[single_feature, :]).cpu()))]
                            #     # print(f'The most valuable feature was {mvp_feature}')
                            #     dl_mvp_features.append(mvp_feature)
                            #     dl_cont_features.append(dl_attr_test[single_feature, :].cpu().tolist())

                            # for single_feature in range(fa_attr_test.shape[0]):
                            #     mvp_feature = temp_bloods.columns[
                            #         int(np.argmax(torch.abs(fa_attr_test[single_feature, :]).cpu()))]
                            #     # print(f'The most valuable feature was {mvp_feature}')
                            #     fa_mvp_features.append(mvp_feature)
                            #     fa_cont_features.append(fa_attr_test[single_feature, :].cpu().tolist())
                            occlusion_count += 1
                test_running_mvp_features.append(mvp_features)
                test_running_cont_features.append(cont_features)
                test_running_ig_mvp_features.append(ig_mvp_features)
                test_running_ig_cont_features.append(ig_cont_features)
                test_running_ignt_mvp_features.append(ignt_mvp_features)
                test_running_ignt_cont_features.append(ignt_cont_features)
                # test_running_dl_mvp_features.append(dl_mvp_features)
                # test_running_dl_cont_features.append(dl_cont_features)
                test_running_fa_mvp_features.append(fa_mvp_features)
                test_running_fa_cont_features.append(fa_cont_features)

    # Test scores
    y_pred = np.array(test_running_val_preds)
    y_true = np.array(test_running_val_labels)
    print(len(y_pred), len(y_true))
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    test_accs = np.array(test_accs)
    test_aucs = np.array(test_aucs)

    # Most important results!
    if do_train:
        print('Val ROC AUC mean:', mean_val_ROC_final, 'std:', std_val_ROC_final)
        print('Val PR AUC mean:', mean_val_PR_final, 'std:', std_val_PR_final)
        print('Val Balanced Accuracy:', mean_val_BA_final, 'std:', std_val_BA_final)
    print('Test AUC mean:', np.mean(test_aucs), 'std:', np.std(test_aucs))
    print('Test Balanced Accuracy mean:', np.mean(test_accs), 'std:', np.std(test_accs))

    print(f'type of test_running_val_names, test_running_val_labels, test_running_val_preds, test_running_mvp_features are {len(test_running_val_names)},'
          f'{len(test_running_val_labels)}, {len(test_running_val_preds)}, {len(test_running_mvp_features)},'
          f'{len(test_running_fa_mvp_features)}, {len(test_running_fa_cont_features)}',
          f'{len(test_running_ignt_mvp_features)}, {len(test_running_ignt_cont_features)}',
          f'{len(test_running_ig_mvp_features)}, {len(test_running_ig_cont_features)}',
          # f'{type(test_running_dl_mvp_features[-1])}, {type(test_running_dl_cont_features[-1])}'
          )
    sub_test = pd.DataFrame(pad_dict_list({"PID": test_running_val_names, "Died": np.squeeze(test_running_val_labels), "Pred": np.squeeze(test_running_val_preds),
                                           "MVP_feat": test_running_mvp_features, 'Cont_feat': test_running_cont_features,
                                           "IG_MVP_feat": test_running_ig_mvp_features, 'IG_Cont_feat': test_running_ig_cont_features,
                                           "IGNT_MVP_feat": test_running_ignt_mvp_features, 'IGNT_Cont_feat': test_running_ignt_cont_features,
                                           # "DL_MVP_feat": test_running_dl_mvp_features, 'DL_Cont_feat': test_running_dl_cont_features,
                                           "FA_MVP_feat": test_running_fa_mvp_features, 'FA_Cont_feat': test_running_fa_cont_features
                                           }))
    sub_test[str(temp_bloods.columns.to_list())] = 0.0
    sub_test.to_csv(os.path.join(SAVE_PATH, 'preds_test.csv'), index=False)


print('END')

time = False
if time:
    # Some latest creation stuff
    tester = pd.read_csv('/data/COVID/Labels/cxr_folds.csv')
    # https://stackoverflow.com/questions/41191365/python-datetime-strptime-error-is-a-bad-directive-in-format-m-d-y-h
    tester.CXR_datetime = pd.to_datetime(tester.CXR_datetime, format="%d/%m/%Y %H:%M")

    DF = pd.DataFrame()
    for id in tester.patient_pseudo_id.unique():
        tmp = tester[tester.patient_pseudo_id == id]
        latest_entry_value = tmp.CXR_datetime.argmax()
        latest_entry = tmp.iloc[latest_entry_value]
        DF = DF.append(latest_entry, ignore_index=True)
    DF.to_csv('/data/COVID/Labels/cxr_folds_latest.csv', index=False, header=True, columns=tester.columns.to_list())
