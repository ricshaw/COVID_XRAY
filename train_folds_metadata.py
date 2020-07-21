import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
#import pydicom
import pandas as pd
from PIL import Image
from PIL.Image import fromarray
from skimage import color
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.preprocessing import StandardScaler

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

        # A transform
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)

        image = self.transform(image)
        label = self.df.Died[index]

        #age = self.df.Age[index]
        #gender = self.df.Gender[index]
        #features = np.stack((age, gender)).astype(np.float32)
        features = self.df.loc[index, '.cLac':'OBS BMI Calculation'].values.astype(np.float32)

        return image, features, filepath, label

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
        n_feats = 44 #2
        hidden1 = 128
        hidden2 = 256
        dropout = 0.3
        self.fc1 = nn.Linear(n_feats, hidden1, bias=True)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.meta = nn.Sequential(self.fc1,
                                  #nn.BatchNorm1d(hidden1),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  self.fc2,
                                  #nn.BatchNorm1d(hidden2),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout)
                                 )

        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=params_dict[encoder][-1]),
                                        nn.Linear(in_features=n_channels_dict[encoder]+hidden2, out_features=1, bias=True)
                                       )
        #self.net = EfficientNet.from_pretrained(encoder, num_classes=1)

    def forward(self, x, features):
        #print('Network 0:', x.shape, features.shape)
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        x = nn.Flatten()(x)

        #features = F.relu(self.fc1(features))
        #features = F.relu(self.fc2(features))
        features = self.meta(features)
        #print('Network 1:', x.shape, features.shape)

        x = torch.cat([x, features], dim=1)
        #print('Network 2:', x.shape, features.shape)

        out = self.classifier(x)
        #out = self.net(x)
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


## Config
labels = '/nfs/home/richard/COVID_XRAY/cxr_news2_pseudonymised_filenames_latest_filled_folds.csv'
encoder = 'efficientnet-b3'
EPOCHS = 100
bs = 32
input_size = (352,352)
FOLDS = 5
alpha = 0.75
gamma = 2.0
OCCLUSION = False
SAVE = True
SAVE_NAME = encoder + '-bs%d-%d-tta-ranger-meta' % (bs, input_size[0])
SAVE_PATH = '/nfs/home/richard/COVID_XRAY/' + SAVE_NAME
print(SAVE_NAME)
log_name = './runs/' + SAVE_NAME
writer = SummaryWriter(log_dir=log_name)

if SAVE:
    os.makedirs(SAVE_PATH, exist_ok=True)


## Load labels
df = pd.read_csv(labels)
print(df.shape)
print(df.head())
print("Number of images:", df.shape[0])
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])

## Remove missing data
#df = df[df.Gender!='Missing']
#df = df[df.Age!=-99]
df = df.replace('Male', 1)
df = df.replace('Female', 0)
#df = df.replace('Chest - X ray', 1)
#df = df.replace('Chest - Xray (Mobile)', 2)
#df = df.replace('Nasogastric Tube Check - X Ray', 3)

## Normalise meta data
#df.Age -= df.Age.mean()
#df.Age /= df.Age.std()
df.Age /= df.Age.max()
#metadata = np.stack((df.Age, df.Gender), axis=1)
#scaler = StandardScaler()
#scaler.fit(metadata)
#metadata = scaler.transform(metadata)
#df.Age = metadata[:,0]
#df.Gender = metadata[:,1]

bloods = df.loc[:,'.cLac':'OBS BMI Calculation'].values.astype(np.float32)
print('Bloods', bloods.shape)
scaler = StandardScaler()
scaler.fit(bloods)
bloods = scaler.transform(bloods)
df.loc[:,'.cLac':'OBS BMI Calculation'] = bloods

## Transforms
A_transform = A.Compose([
                         A.Flip(p=1),
                         A.RandomRotate90(p=1),
                         A.Rotate(p=1, limit=45, interpolation=3),
                         A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.7,1.0), ratio=(0.8,1.2), interpolation=3, p=1),
                         A.OneOf([
                                  A.IAAAdditiveGaussianNoise(),
                                  A.GaussNoise(),
                                 ], p=0.3),
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
                                 ], p=0.05),
                         A.OneOf([
                                  A.CLAHE(clip_limit=2),
                                  A.IAASharpen(),
                                  A.IAAEmboss(),
                                  A.RandomBrightnessContrast(),
                                 ], p=1),
                         A.RandomGamma(p=1),
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
    val_loader = DataLoader(val_dataset, batch_size=int(bs/2), num_workers=4)

    ## Init model
    model = Model(encoder)
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
            out = model(images, features)
            #loss = criterion(out, labels)
            loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total += labels.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.5).int() == labels).sum().item()
            print("iter: {}, Loss: {}".format(i, loss.item()) )

            res_prob += out.detach().cpu().numpy().tolist()
            res_label += labels.detach().cpu().numpy().tolist()

        y_true = np.array(res_label)
        y_scores = np.array(res_prob)
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        # Writing to tensorboard
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
        writer.add_image('images', grid, epoch)
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('AUC/train', auc, epoch)
        writer.add_scalar('AP/train', ap, epoch)

        print("Epoch: {}, Loss: {}, Train Accuracy: {}".format(epoch, running_loss, round(correct/total, 4)))
        #if epoch % 2 == 1:
        #    scheduler.step()

        # Save model
        if SAVE:
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
                print('images', images.shape, 'features', features.shape, 'labels', labels.shape)

                ## TTA
                batch_size, n_crops, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                _, n_feats = features.size()
                features = features.repeat(1,n_crops).view(-1,n_feats)
                print('tta images', images.shape, features.shape)
                out = model(images, features)
                out = out.view(batch_size, n_crops, -1).mean(1)

                #loss = criterion(out.data, labels)
                loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")

                running_loss += loss.item()

                total += labels.size(0)
                out = torch.sigmoid(out)
                correct += ((out > 0.5).int() == labels).sum().item()

                res_prob += out.cpu().numpy().tolist()
                res_label += labels.cpu().numpy().tolist()

                if epoch == (EPOCHS-1):
                    val_preds += out.cpu().numpy().tolist()
                    val_labels += labels.cpu().numpy().tolist()
                    val_names += names

                if count == 0:
                    images = images.view(batch_size, n_crops, -1)[:,0,...]
                    oc_images = images[(labels==1).squeeze()].cuda()
                    oc_labels = labels[(labels==1).squeeze()].cuda()
                count += 1

        acc = correct/total
        y_true = np.array(res_label)
        y_scores = np.array(res_prob)
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        running_auc.append(auc)
        print('ALL AUCs', running_auc)
        print('Best AUC', np.argmax(running_auc))
        if epoch == (EPOCHS-1):
            val_auc.append(auc)

        writer.add_scalar('Loss/val', loss.item(), epoch)
        writer.add_scalar('AUC/val', auc, epoch)
        writer.add_scalar('AP/val', ap, epoch)

        print("Epoch: {}, Loss: {}, Test Accuracy: {}, AUC: {}".format(epoch, running_loss, round(acc, 4), auc))


        ## Occlusion
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
print('Labels', len(val_labels), 'Preds', len(val_preds), 'AUCs', len(val_auc))
correct = ((val_preds > 0.5).astype(int) == val_labels).sum()
acc = correct / len(val_labels)
auc = roc_auc_score(val_labels, val_preds)
print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), auc))
print('AUC mean:', np.mean(val_auc), 'std:', np.std(val_auc))

res_prob = [x[0] for x in res_prob]
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels.tolist(), "Pred":val_preds.tolist()})
sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

## Plot
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


average_precision = average_precision_score(val_labels, val_preds)
precision, recall, thresholds = precision_recall_curve(val_labels, val_preds)
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=lw, label='AP = %0.2f' % average_precision)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Prediction of Death - Precision-Recall')
plt.legend(loc="lower right")
plt.savefig('precision-recall-' + SAVE_NAME + '.png', dpi=300)


val_labels = [x[0] for x in val_labels]
val_preds = [x[0] for x in val_preds]
sub = pd.DataFrame({"Filename":val_names, "Died":val_labels, "Pred":val_preds})
sub.to_csv('preds-' + SAVE_NAME + '.csv', index=False)
print('END')
