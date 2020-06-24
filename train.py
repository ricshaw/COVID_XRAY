import numpy as np
import matplotlib.pyplot as plt
import os
#import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    #img = Image.open(path).convert('L')
    #img = np.array(img)
    #img = np.clip(img, np.percentile(img,5), np.percentile(img,95))
    #img -= img.min()
    #img /= img.max()
    #img -= np.mean(img)
    #img /= (np.std(img) + 1e-9)
    #img = Image.fromarray(np.uint8(img))
    return img

def dicom_image_loader(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img = np.uint8(255.0*img)
    img = Image.fromarray(img).convert("RGB")
    return img

class ImageDataset(Dataset):
    def __init__(self, img_dir, df, transform):
        self.img_dir = img_dir
        self.df = df
        self.loader = default_image_loader
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.df.Filename[index]
        filepath = os.path.join(self.img_dir, image_name + '.jpg')
        image = self.loader(filepath)
        image = self.transform(image)
        label = self.df.Died[index]
        return image, filepath, label

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
        #self.net = EfficientNet.from_pretrained(encoder)
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = EfficientNet.from_pretrained(encoder, num_classes=1)

    def forward(self, x):
        #x = self.net.extract_features(x)
        #x = self.avg_pool(x)
        #out = nn.Flatten()(x)
        out = self.net(x)
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


img_dir = '/nfs/project/richard/COVID/KCH_CXR_JPG'
labels = 'KCH_CXR_JPG.csv'
encoder = 'efficientnet-b0'
EPOCHS = 11
bs = 128
input_size = (256,256)
SAVE = False
SAVE_PATH = ''

# Load labels
df = pd.read_csv(labels)
print(df.shape)
print(df.head())

# Remove any missing labels
df0 = df[df.Died == 0]
df1 = df[df.Died == 1]
df = df0
df = df.append(df1, ignore_index=True)
print(df0.shape, df1.shape, df.shape)

print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])

# Train / Val split
train_df, val_df = train_test_split(df, stratify=df.Died, test_size=0.10)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                              transforms.Resize(input_size,3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])



train_dataset = ImageDataset(img_dir, train_df, transform)
train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)

val_dataset = ImageDataset(img_dir, val_df, transform)
val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=8)


model = Model(encoder)
use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')
model = nn.DataParallel(model)

#criterion = nn.BCEWithLogitsLoss()
criterion = FocalLoss(logits=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)


model.cuda()
print('\nStarting training!')
for epoch in range(EPOCHS):

    print('Training step')
    running_loss = 0.0
    model.train()
    correct = 0
    total = 0

    for i, sample in enumerate(train_loader):
        images, names, labels = sample[0], sample[1], sample[2]
        #print(images.shape, labels.shape)
        images = images.cuda()
        labels = labels.cuda()

        out = model(images)

        labels = labels.unsqueeze(1).float()
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        total += labels.size(0)
        out = torch.sigmoid(out)
        correct += ((out > 0.5).int() == labels).sum().item()

        print("iter: {}, Loss: {}".format(i, loss.item()) )

    print("Epoch: {}, Loss: {}, Train Accuracy: {}".format(epoch, running_loss, round(correct/total, 4)))
    if epoch % 2 == 1:
        scheduler.step()

    # Save model
    if SAVE:
        MODEL_PATH = os.path.join(SAVE_PATH, ('fold_%d_epoch_%d.pth' % (fold, epoch)))
        print(MODEL_PATH)
        torch.save(model.state_dict(), MODEL_PATH)


    print('Validation step')
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    res_id = []
    res_prob = []
    res_label = []
    with torch.no_grad():
        for images, names, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()
            labels = labels.unsqueeze(1).float()
            out = model(images)
            loss = criterion(out.data, labels)

            running_loss += loss.item()

            total += labels.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.5).int() == labels).sum().item()

            res_id += names
            res_prob += out.cpu().numpy().tolist()
            res_label += labels.cpu().numpy().tolist()

    acc = correct/total
    y_true = np.array(res_label)
    y_scores = np.array(res_prob)
    auc = roc_auc_score(y_true, y_scores)
    print("Epoch: {}, Loss: {}, Test Accuracy: {}, AUC: {}\n".format(epoch, running_loss, round(acc, 4), auc))


print('END')
