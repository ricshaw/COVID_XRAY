from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
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

parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--labels', type=str)
parser.add_argument('--images_dir', type=str)
parser.add_argument('--save_path', type=str)
arguments = parser.parse_args()

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
    def __init__(self, df):
        self.df = df
        self.loader = default_image_loader

    def __getitem__(self, index):
        filepath = self.df.Filename[index]
        # This produces a string of a list
        label = self.df['Died'][index]
        # Convert to int/ float list
        # label = eval(label)

        # Attempt 2: Full
        bloods = self.df[self.df.columns.difference(self.df.filter(like='ICU').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='date of death').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='OHE').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='stratify').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='fold').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='Death').columns, sort=False)]
        bloods = bloods[bloods.columns.difference(bloods.filter(like='Died').columns, sort=False)]
        bloods = bloods.select_dtypes(include=[np.number])
        bloods = np.array(bloods.iloc[index])  # .astype(np.double)
        return filepath, label, bloods

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
img_dir = arguments.images_dir  # '/data/COVID/Data/KCH_CXR_JPG_latest'  # arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
labels = arguments.labels  # '/data/COVID/Labels/KCH_CXR_JPG_latest.csv'   # arguments.labels  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(labels)
# SAVE_PATH = os.path.join(f'/data/COVID/models/{arguments.job_name}')
SAVE_PATH = '/data/COVID/models/death-time-b3-focal-occ-sparse'  # Old focal loss
SAVE_PATH = '/data/COVID/models/death-time-b3-multi-focal-fb'
SAVE_PATH = arguments.save_path  # '/data/COVID/models/death-time-b3-multi-focal-fb'

# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)

# Load labels
print(f'The  labels are {labels}')
# img_dir = img_dir[0]
# labels = labels[0]
df = pd.read_csv(labels)

if 'OHE_Time_To_Death' not in df.columns:
    filenames = df['Filename']
    death_dates = df['Death_DTM']
    time_differences = []
    for ID, filename in enumerate(filenames):
        scan_time = filename.split('_')[1]
        scan_date = datetime.datetime(year=int(scan_time[0:4]),
                                      month=int(scan_time[4:6]),
                                      day=int(scan_time[6:8]))
        if death_dates[ID] == '00/01/1900':
            time_string = '01/01/1900'
        else:
            time_string = death_dates[ID]
        death_date = datetime.datetime.strptime(time_string, "%d/%m/%Y")
        time_difference = abs((death_date - scan_date).days)
        time_differences.append(time_difference)
    df['Time_To_Death'] = time_differences

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

# Append image directory to filename field
if '.' in df['Filename'][0][-1]:
    df['Filename'] = img_dir + '/' + df['Filename'].astype(str)
else:
    df['Filename'] = img_dir + '/' + df['Filename'].astype(str) + '.jpg'

# For shape purposes:
temp_bloods = df[df.columns.difference(df.filter(like='ICU').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='date of death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='OHE').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='stratify').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='fold').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Died').columns, sort=False)]
temp_bloods = temp_bloods.select_dtypes(include=[np.number])

print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])


# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        n_feats = len(temp_bloods.columns)
        hidden1 = 128
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

        self.classifier = nn.Linear(hidden2, out_features=1, bias=True)

    def forward(self, features):
        features = self.meta(features)
        out = self.classifier(features)
        # out = self.net(x)
        return out


running_loss = 0
# correct = 0
val_counter = 0
total = 0
res_prob = []
res_label = []
res_name = []
class_names = ['Died', 'micro']
# Hyperparameter loading
model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))

for model_file in model_files:
    # Train / Val split
    # train_df, val_df = train_test_split(df, stratify=df.Died, test_size=0.10, random_state=37)
    if 'fold_0' in model_file:
        fold = 0
    elif 'fold_1' in model_file:
        fold = 1
    elif 'fold_2' in model_file:
        fold = 2
    elif 'fold_3' in model_file:
        fold = 3
    elif 'fold_4' in model_file:
        fold = 4
    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    print(f'The length of the training is {len(train_df)}')
    print(f'The length of the validation is {len(val_df)}')

    latest_model_file = model_file
    # checkpoint = torch.load(latest_model_file, map_location={'cuda:0': 'cpu'})
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cpu'))
    print(f'Loading {latest_model_file}, fold {fold}')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    # Extras that may not exist in older models
    bs = checkpoint['batch_size']
    EPOCHS = 170

    model = Model()
    use_cuda = torch.cuda.is_available()
    print('Using cuda', use_cuda)

    if use_cuda and torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run model in eval mode:
    model.eval()
    val_dataset = ImageDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=int(bs / 4), num_workers=8)

    with torch.no_grad():
        for names, labels, bloods in val_loader:
            labels = labels.cuda()
            labels = labels.unsqueeze(1).float()
            bloods = bloods.cuda()
            bloods = bloods.float()

            out = model(bloods)

            out = torch.sigmoid(out)

            total += labels.numel()
            # out = torch.sigmoid(out)

            acc = ((out > 0.5).int() == labels).sum().item()
            # correct += ((out > 0.5).int() == labels).sum().item()

            res_name += names
            res_prob += out.cpu().numpy().tolist()
            res_label += labels.cpu().numpy().tolist()
            val_counter += 1

# acc = correct / total
# class_correct = [i*num_classes/total for i in class_correct]
y_true = np.array(res_label)
y_scores = np.array(res_prob)
true_auc = roc_auc_score(y_true, y_scores)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute PR curve and PR area for each class
precision_tot = dict()
recall_tot = dict()
pr_auc = dict()

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

# # Save csvs with results
print(len(res_name), len(y_true), len(y_scores))
sub = pd.DataFrame({"Filename": res_name, "Died": y_true.tolist(), "Pred": y_scores.tolist()})
sub_name = 'preds.csv'
sub.to_csv(os.path.join(SAVE_PATH, sub_name), index=False)
