import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2
import sys
import random
import pandas as pd
from collections import OrderedDict
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

from efficientnet_pytorch import EfficientNet

sys.path.append('/nfs/home/richard/over9000')
from over9000 import RangerLars

sys.path.append('/nfs/home/richard/A-journey-into-Convolutional-Neural-Network-visualization-/visualisation/core')
from GradCam import GradCam
from utils import device
from utils import image_net_postprocessing
from utils import *
from utils import image_net_preprocessing


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

KCH_data = '/nfs/home/richard/COVID_XRAY/cxr_news2_pseudonymised_filenames_folds.csv'
GSTT_data = '/nfs/home/richard/COVID_XRAY/new_gstt_folds.csv'

## Load KCH data
df1 = pd.read_csv(KCH_data)
df1 = df1.sort_values(by=['patient_pseudo_id'], ascending=True)
df1['Filename'] = df1['Filename'].astype(str)
df1 = df1[['patient_pseudo_id','Filename','Died','fold']].reset_index(drop=True)
print('KCH data:', df1.shape)
print(df1.head(20))

## Load GSTT data
df2 = pd.read_csv(GSTT_data)
df2 = df2.dropna(subset=['Filename'])
df2['Filename'] = '/nfs/home/pedro/COVID/GSTT_JPGs_All/' + df2['Filename'].astype(str)
df2['patient_pseudo_id'] = df2['patient_pseudo_id'].astype(str)
df2 = df2[df2.Time_Mismatch < 2]
df2 = df2[['patient_pseudo_id','Filename','Died','fold']].reset_index(drop=True)
print('GSTT data:', df2.shape)
print(df2.head(20))

## Combine KCH and GSTT
df = pd.concat([df1,df2]).reset_index(drop=True)
print('All data:', df.shape)
print('Number of images:', df.shape[0])
print('Died:', df[df.Died == 1].shape[0])
print('Survived:', df[df.Died == 0].shape[0])
print(df.head(20))


def new_section(text):
    print('\n')
    print(100 * '*')
    print(text)

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

CENTRE_CROP = True
def get_image(df, idx, transform, A_transform=None):
    image = default_image_loader(df.Filename[idx])
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


def tensor2img(tensor, ax=plt):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img

class ImageDataset(Dataset):
    def __init__(self, my_df, transform, A_transform=None, mode=None):
        self.df = my_df
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform
        self.mode = mode

    def __getitem__(self, index):
        pid = self.df.patient_pseudo_id[index]
        image = get_image(self.df, index, self.transform, self.A_transform)
        if self.mode=='test':
            return pid, image
        else:
            label = self.df.Died[index]
            return pid, image, label

    def __len__(self):
        return self.df.shape[0]

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

class Model(nn.Module):
    def __init__(self, encoder='efficientnet-b0'):
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
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),}
        self.out_chns = 0
        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out_chns += n_channels_dict[encoder]
        self.fc = nn.Linear(self.out_chns, 1)
        self.dropouts = nn.ModuleList([nn.Dropout(config['dropout']) for _ in range(5)])

    def forward(self, image):
        x = self.net.extract_features(image)
        x = self.avg_pool(x)
        x = nn.Flatten()(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        out /= len(self.dropouts)
        return out



ROOT = '/nfs/home/richard/COVID_XRAY'
MODEL_NAME = 'image-model-bs8-lr0.001-dp0.1-efficientnet-b5-sz448'
ENCODER = MODEL_NAME.split('-')[-3] + '-' + MODEL_NAME.split('-')[-2]
BATCH_SIZE = int(MODEL_NAME.split('-')[2][2::])
LR = float(MODEL_NAME.split('-')[3][2::])
DROPOUT = float(MODEL_NAME.split('-')[-4][2::])
IMAGE_SIZE = int(MODEL_NAME.split('-')[-1][2::])

config = dict(  batch_size=BATCH_SIZE,
                lr=LR,
                dropout=DROPOUT,
                image_size=IMAGE_SIZE,
                encoder=ENCODER,)
print(config)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = transforms.Compose([
                              transforms.Resize(config['image_size'], 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])

## Get inputs
inds = range(df.shape[0])
#inds = range(2000)
names = df.Filename[inds]

inputs = [get_image(df, i, val_transform, A_transform=None).unsqueeze(0) for i in inds]
inputs = [x.to(device) for x in inputs]
for x in inputs:
    print('in:', x.shape)


## Init model
model = Model(config['encoder']).cuda()
model = nn.DataParallel(model)


## Load model
fold = 0
MODEL_PATH = os.path.join(ROOT, MODEL_NAME)
MODEL_PATH = os.path.join(MODEL_PATH, ('fold_%d_best.pth' % (fold)))
print(MODEL_PATH)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print('Loaded:', MODEL_PATH)

## GradCam
model.eval()
vis = GradCam(model, 'cuda')
model_outs = list(map(lambda x: tensor2img(vis(x, None, postprocessing=image_net_postprocessing)[0]), inputs))


## Get outputs
cnt = 0
for out in model_outs:
    print('out:', out.shape, out.min(), out.max())
    out -= out.min()
    out /= out.max()
    out *= 255.0
    SAVE_NAME = os.path.join(ROOT, MODEL_NAME)
    name = names[cnt].split('/')[-1]
    SAVE_NAME = os.path.join(SAVE_NAME, 'gradcam_' + name)
    print(SAVE_NAME)
    cv2.imwrite(SAVE_NAME, out.astype(np.uint8)[...,::-1])
    cnt += 1


## Clear up
del model
torch.cuda.empty_cache()
