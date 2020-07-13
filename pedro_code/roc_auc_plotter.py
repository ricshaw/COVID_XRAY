from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
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
import matplotlib
matplotlib.use('TkAgg')


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.cuda.empty_cache()
# Load the model and the val_dataset
# parser = argparse.ArgumentParser(description='Passing files + relevant directories')
# parser.add_argument('--labels', nargs='+', type=str)
# parser.add_argument('--images_dir', nargs='+', type=str)
# parser.add_argument('--job_name', type=str)
# parser.add_argument('--mode', type=str)
# arguments = parser.parse_args()


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
        # Augmentations
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)
        image = self.transform(image)
        label = self.df['OHE_Time_To_Death'][index]
        label = np.array(label)
        return image, filepath, label

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


# Some necessary variables
img_dir = '/data/COVID/Data/KCH_CXR_JPG_latest'  # arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
labels = '/data/COVID/Labels/KCH_CXR_JPG_latest.csv'   # arguments.labels  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(labels)
# SAVE_PATH = os.path.join(f'/data/COVID/models/{arguments.job_name}')
SAVE_PATH = '/data/COVID/models/death-time-b3-focal-occ-sparse'  # Old focal loss
SAVE_PATH = '/data/COVID/models/death-time-b3-multi-focal-fb'

# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)

# Hyperparameter loading
model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
latest_model_file = max(model_files, key=os.path.getctime)
# checkpoint = torch.load(latest_model_file, map_location={'cuda:0': 'cpu'})
checkpoint = torch.load(latest_model_file, map_location=torch.device('cpu'))
print(f'Loading {latest_model_file}')
encoder = checkpoint['encoder']
loaded_epoch = checkpoint['epoch']
loss = checkpoint['loss']
running_iter = checkpoint['running_iter']
# Extras that may not exist in older models
bs = checkpoint['batch_size']
input_size = checkpoint['resolution']
EPOCHS = 1000

# Load labels
print(f'The  labels are {labels}')
# img_dir = img_dir[0]
# labels = labels[0]
df = pd.read_csv(labels)
filenames = df['Filename']
death_dates = df['Death_DTM']
time_differences = []
for ID, filename in enumerate(filenames):
    scan_time = filename.split('_')[1]
    scan_date = datetime.datetime(year=int(scan_time[0:4]),
                                  month=int(scan_time[4:6]),
                                  day=int(scan_time[6:8]))
    death_date = datetime.datetime.strptime(death_dates[ID], "%d/%m/%Y")
    time_difference = abs((death_date - scan_date).days)
    time_differences.append(time_difference)
df['Time_To_Death'] = time_differences
df['Filename'] = img_dir + '/' + df['Filename'].astype(str)

# OHE labels
ohe_labels = []
cutoffs = [2, 7, 100, 1e50]
num_classes = len(cutoffs)
for time_to_death_label in df['Time_To_Death']:
    ohe_label = [0, 0, 0, 0]
    # Find out which range the time belongs to by finding index of first truth
    time_class = [time_to_death_label < cutoff for cutoff in cutoffs].index(True)
    ohe_label[time_class] = 1.0
    ohe_labels.append(ohe_label)

# Add to dataframe
df['OHE_Time_To_Death'] = ohe_labels


# Exclude all entries with "Missing" Died stats
df = df[~df['Died'].isin(['Missing'])]
df['Died'] = pd.to_numeric(df['Died'])

# Augmentations
A_transform = A.Compose([
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.Rotate(p=1, limit=45, interpolation=3),
                         A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.8,1.0), ratio=(0.8,1.2), interpolation=3, p=1),
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

print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])

# Train / Val split
train_df, val_df = train_test_split(df, stratify=df.Died, test_size=0.10, random_state=37)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print(f'The length of the training is {len(train_df)}')
print(f'The length of the validation is {len(val_df)}')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Pre-processing transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
      transforms.Resize(input_size, 3),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
])

val_dataset = ImageDataset(val_df, val_transform)
val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=8)

# Model
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
        # self.net = EfficientNet.from_pretrained(encoder)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = EfficientNet.from_pretrained(encoder, num_classes=num_classes)

    def forward(self, x):
        # x = self.net.extract_features(x)
        # x = self.avg_pool(x)
        # out = nn.Flatten()(x)
        out = self.net(x)
        return out


model = Model(encoder)
use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'])


# Run model in eval mode:
num_classes = 4
model.eval()
running_loss = 0
# correct = 0
class_correct = [0] * num_classes
val_counter = 0
total = 0
res_id = []
res_prob = []
res_label = []
class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']

with torch.no_grad():
    for images, names, labels in val_loader:
        images = images
        labels = labels
        labels = labels.float()
        out = model(images)
        out = torch.softmax(out, dim=1)
        # val_loss = criterion(out.data, labels)

        # running_loss += val_loss.item()

        # total += labels.numel()
        # out = torch.sigmoid(out)

        # for classID in range(num_classes):
        #     acc = ((out[:, classID] > 0.5).int() == labels[:, classID]).sum().item()
        #     acc = round(acc, 4)
        #     class_correct[classID] += acc
        # correct += ((out > 0.5).int() == labels).sum().item()

        res_id += names
        res_prob += out.cpu().numpy().tolist()
        res_label += labels.cpu().numpy().tolist()
        print(val_counter)
        val_counter += 1

# acc = correct / total
# class_correct = [i*num_classes/total for i in class_correct]
y_true = np.array(res_label)
y_scores = np.array(res_prob)
true_auc = roc_auc_score(y_true, y_scores)
class_auc = []
for classID in range(num_classes):
    auc_score = roc_auc_score(y_true[:, classID], y_scores[:, classID])
    class_auc.append(auc_score)
class_pr = []
for classID in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true[:, classID], y_scores[:, classID])
    pr_auc = auc(recall, precision)
    class_pr.append(pr_auc)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for classID in range(num_classes):
    fpr[classID], tpr[classID], _ = roc_curve(y_true[:, classID], y_scores[:, classID])
    roc_auc[classID] = auc(fpr[classID], tpr[classID])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute PR curve and PR area for each class
precision_tot = dict()
recall_tot = dict()
pr_auc = dict()
for classID in range(num_classes):
    precision_tot[classID], recall_tot[classID], _ = precision_recall_curve(y_true[:, classID], y_scores[:, classID])
    pr_auc[classID] = auc(recall_tot[classID], precision_tot[classID])

# Compute micro-average precision-recall curve and PR area
precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])
no_skill = len(y_true[y_true == 1]) / len(y_true)

colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red']
# Plot ROC-AUC for different classes:
plt.figure()
plt.axis('square')
for classID, key in enumerate(fpr.keys()):
    lw = 2
    plt.plot(fpr[key], tpr[key], color=colors[classID],  # 'darkorange',
             lw=lw, label=f'{class_names[classID]} ROC curve (area = {roc_auc[key]: .2f})')
    plt.title(f'Class ROC-AUC for ALL classes', fontsize=18)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
# lw = 2
# plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
# plt.title('Class ROC-AUC for ALL classes')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")

plt.figure()
plt.axis('square')
for classID, key in enumerate(precision_tot.keys()):
    lw = 2
    plt.plot(recall_tot[key], precision_tot[key], color=colors[classID],  # color='darkblue',
             lw=lw, label=f'{class_names[classID]} PR curve (area = {pr_auc[key]: .2f})')
    plt.title(f'Class PR-AUC for ALL classes', fontsize=18)
    # plt.plot([0, 1], [0, 0], lw=lw, linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower right")
plt.show()
# plt.figure()
# plt.plot(recall_tot["micro"], precision_tot["micro"], color='darkblue',
#          lw=lw, label='PR curve (area = %0.2f)' % pr_auc["micro"])
# plt.title('Class PR-AUC for ALL classes')
# # plt.plot([0, 1], [0, 0], lw=lw, linestyle='--', label='No Skill')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc="lower right")
# plt.show()
