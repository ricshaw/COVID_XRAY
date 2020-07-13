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
from torch.utils.tensorboard import SummaryWriter

# Load labels
labels = '/data/COVID/Labels/KCH_CXR_JPG_latest.csv'
img_dir = '/data/COVID/Data/KCH_CXR_JPG_latest'
print(f'The  labels are {labels}')
labels = labels
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
Enc_labels = []
cutoffs = [2, 7, 100, 1e50]
num_classes = len(cutoffs)
for time_to_death_label in df['Time_To_Death']:
    ohe_label = [0, 0, 0, 0]
    # Find out which range the time belongs to by finding index of first truth
    time_class = [time_to_death_label < cutoff for cutoff in cutoffs].index(True)
    Enc_labels.append(time_class)
    ohe_label[time_class] = 1.0
    ohe_labels.append(ohe_label)

# Add to dataframe
df['OHE_Time_To_Death'] = ohe_labels
df['Enc_Time_To_Death'] = Enc_labels

# Exclude all entries with "Missing" Died stats
df = df[~df['Died'].isin(['Missing'])]
df['Died'] = pd.to_numeric(df['Died'])

df.to_csv('/data/COVID/Labels/KCH_CXR_JPG_latest_dt.csv', index=False)

# Mobile vs Non-mobile
mobiles = df[df.Examination_Title == 'Chest - Xray (Mobile)']
non_mobiles = df[df.Examination_Title == 'Chest - X ray']

plt.figure(1)
plt.title('Mobiles')
plt.hist(mobiles.Enc_Time_To_Death)
plt.figure(2)
plt.title('Non-Mobiles')
plt.hist(non_mobiles.Enc_Time_To_Death)
plt.show()
