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
import sys
sys.path.append('/nfs/home/pedro/RangerLARS/over9000')
from over9000 import RangerLars
sys.path.append('/nfs/home/pedro/ImbalancedDatasetSampler/imbalanced-dataset-sampler/torchsampler')
from imbalanced import ImbalancedDatasetSampler

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
        # Augmentations
        if self.A_transform is not None:
            image = np.array(image)
            image = self.A_transform(image=image)['image']
            image = Image.fromarray(image)
        image = self.transform(image)
        image = np.array(image)
        # This produces a string of a list
        label = self.df['OHE_Time_To_Death'][index]
        # Convert to int/ float list
        label = eval(label)
        label = np.array([int(ohe) for ohe in label])
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


# Some necessary variables
img_dir = arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
labels = arguments.labels  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
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
    encoder = checkpoint['encoder']
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    # Extras that may not exist in older models
    bs = checkpoint['batch_size']
    input_size = checkpoint['resolution']
    EPOCHS = 100
    FOLDS = 5
else:
    running_iter = 0
    loaded_epoch = -1
    bs = 32
    input_size = (512, 512)  # (528, 528)
    encoder = 'efficientnet-b3'
    EPOCHS = 100
    FOLDS = 5

# Load labels
print(f'The  labels are {labels}')
num_classes = 4
if len(labels) == 1:
    img_dir = img_dir[0]
    labels = labels[0]
    df = pd.read_csv(labels)
    df['Filename'] = img_dir + '/' + df['Filename'].astype(str) + '.jpg'

elif len(labels) > 1:
    df = pd.read_csv(labels[0])
    df['Filename'] = img_dir[0] + df['Filename'].astype(str)
    for extra in range(1, len(labels)):
        extra_df = pd.read_csv(labels[extra])
        extra_df['Filename'] = img_dir[extra] + '/' + extra_df['Filename']  # .astype(str)
        df = pd.concat([df, extra_df], ignore_index=True)

# # Exclude all entries with "Missing" Died stats
# df = df[~df['Died'].isin(['Missing'])]
# df['Died'] = pd.to_numeric(df['Died'])

# Pre-processing transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
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
                         A.RandomResizedCrop(input_size[0], input_size[1], scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=3, p=1),
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

use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')

# For aggregation
val_preds = []
val_labels = []
val_names = []
overall_val_roc_aucs = []
overall_val_pr_aucs = []
class_val_roc_aucs = []
class_val_pr_aucs = []

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
    # Pre-loading sequence
    model = Model(encoder)
    alpha = torch.FloatTensor([0.979, 0.931, 0.811, 0.279])[None, ...].cuda()
    alpha = alpha ** 0.5
    criterion = FocalLoss(logits=True)
    # Try RAdam + Lookahead
    optimizer = RangerLars(model.parameters())
    # Outdated
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    # Specific fold writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold}'))

    # for name, child in model.net.named_parameters():
    #     print(name)

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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Get the validation entries from previous folds!
        val_preds = checkpoint['val_preds']
        val_labels = checkpoint['val_labels']
        overall_val_roc_aucs = checkpoint['overall_val_roc_aucs']
        overall_val_pr_aucs = checkpoint['overall_val_pr_aucs']
        class_val_roc_aucs = checkpoint['class_val_roc_aucs']
        class_val_pr_aucs = checkpoint['class_val_pr_aucs']
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
    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=ImbalancedDatasetSampler(train_dataset),
                              num_workers=8)  #, shuffle=True)

    val_dataset = ImageDataset(val_df, tta_transform)
    val_loader = DataLoader(val_dataset, batch_size=int(bs/4), num_workers=8)

    # Training
    if arguments.mode == 'train' or arguments.mode == 'pretrained':
        model.cuda()
        print('\nStarting training!')
        for epoch in range(latest_epoch+1, EPOCHS):
            print('Training step')
            running_loss = 0.0
            model.train()
            train_class_correct = [0] * num_classes
            total = 0

            for i, sample in enumerate(train_loader):
                images, names, labels = sample[0], sample[1], sample[2]
                # print(images.shape, labels.shape)
                images = images.cuda()
                labels = labels.cuda()

                out = model(images)
                # out = torch.softmax(out, dim=1)
                # print(out.shape, out)

                labels = labels.float()
                loss = criterion(out, labels)
                out = torch.softmax(out, dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                total += labels.numel()
                # out = torch.sigmoid(out)
                for classID in range(num_classes):
                    train_acc = ((out[:, classID] > 0.5).int() == labels[:, classID]).sum().item()
                    train_acc = round(train_acc, 4)
                    train_class_correct[classID] += train_acc

                # correct += ((out > 0.5).int() == labels).sum().item()

                # Name check: Shuffling sanity check
                if i == 0:
                    print(f'The test names are: {names[0]}, {names[-2]}')

                # Convert labels and output to grid
                images_grid = torchvision.utils.make_grid(images)
                labels_grid = torchvision.utils.make_grid(labels)
                rounded_output_grid = torchvision.utils.make_grid((out > 0.5).int())
                output_grid = torchvision.utils.make_grid(out)

                # Writing to tensorboard
                if running_iter % 50 == 0:
                    writer.add_scalar('Loss/train', loss.item(), running_iter)
                    writer.add_image('Visuals/Images', image_normaliser(images_grid), running_iter)
                    writer.add_image('Visuals/Labels', image_normaliser(labels_grid), running_iter)
                    writer.add_image('Visuals/Rounded Output', image_normaliser(rounded_output_grid), running_iter)
                    writer.add_image('Visuals/Output', image_normaliser(output_grid), running_iter)

                print("iter: {}, Loss: {}".format(running_iter, loss.item()))
                running_iter += 1

            train_class_correct = [t*num_classes/total for t in train_class_correct]
            print("Epoch: {}, Loss: {},\n Train Accuracy: {}".format(epoch, running_loss, train_class_correct))
            # if epoch % 2 == 1:
            #     scheduler.step()

            print('Validation step')
            model.eval()
            running_loss = 0
            # correct = 0
            class_correct = [0] * num_classes
            val_counter = 0
            total = 0
            res_id = []
            res_prob = []
            res_label = []
            if epoch == (EPOCHS - 1):
                occlusion = True
            else:
                occlusion = False
            if occlusion:
                for images, names, labels in val_loader:
                    # Pick one image
                    random_index = np.random.randint(images.size(0))
                    images = images[random_index, ...][None, ...].cuda()
                    names = names[random_index]
                    name = os.path.basename(names)
                    name = os.path.splitext(name)[0]
                    labels = labels[random_index, ...][None, ...].cuda()
                    # print(label.shape, label)
                    # print(image.shape, image)
                    print(images.shape, labels.shape)
                    images = images.cuda()
                    labels = labels.cuda()

                    # Account for tta: Take first image (non-augmented)
                    # Label does not need to be touched because it is obv. the same for this image regardless of tta
                    images = images[:, 0, ...]

                    # Set a baseline
                    baseline = torch.zeros_like(images).cuda()

                    # Calculate attribution scores + delta
                    # ig = IntegratedGradients(model)
                    oc = Occlusion(model)
                    # nt = NoiseTunnel(ig)
                    # attributions, delta = nt.attribute(image, nt_type='smoothgrad', stdevs=0.02, n_samples=2,
                    #                                    baselines=baseline, target=0, return_convergence_delta=True)
                    _, target_ID = torch.max(labels, 1)
                    print(target_ID)
                    # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)
                    x_shape = 16
                    x_stride = 8

                    oc_attributions0 = oc.attribute(images, sliding_window_shapes=(3, x_shape, x_shape),
                                                    strides=(3, x_stride, x_stride), target=0,
                                                    baselines=baseline)
                    oc_attributions1 = oc.attribute(images, sliding_window_shapes=(3, x_shape, x_shape),
                                                    strides=(3, x_stride, x_stride), target=1,
                                                    baselines=baseline)
                    oc_attributions2 = oc.attribute(images, sliding_window_shapes=(3, x_shape, x_shape),
                                                    strides=(3, x_stride, x_stride), target=2,
                                                    baselines=baseline)
                    oc_attributions3 = oc.attribute(images, sliding_window_shapes=(3, x_shape, x_shape),
                                                    strides=(3, x_stride, x_stride), target=3,
                                                    baselines=baseline)
                    # print('IG + SmoothGrad Attributions:', attributions)
                    # print('Convergence Delta:', delta)

                    # Write to tensorboard
                    image_grid = torchvision.utils.make_grid(images)
                    # attributions_grid = torchvision.utils.make_grid(torch.abs(attributions))
                    oc_attributions_grid0 = torchvision.utils.make_grid(torch.abs(oc_attributions0))
                    oc_attributions_grid1 = torchvision.utils.make_grid(torch.abs(oc_attributions1))
                    oc_attributions_grid2 = torchvision.utils.make_grid(torch.abs(oc_attributions2))
                    oc_attributions_grid3 = torchvision.utils.make_grid(torch.abs(oc_attributions3))

                    # Write to tensorboard
                    writer.add_image('Interpretability/Image', image_normaliser(image_grid), running_iter)
                    # writer.add_image('Interpretability/Attributions', image_normaliser(attributions_grid), running_iter)
                    writer.add_image('Interpretability/OC_Attributions_48H', image_normaliser(oc_attributions_grid0), running_iter)
                    writer.add_image('Interpretability/OC_Attributions_under_week', image_normaliser(oc_attributions_grid1), running_iter)
                    writer.add_image('Interpretability/OC_Attributions_over_week', image_normaliser(oc_attributions_grid2), running_iter)
                    writer.add_image('Interpretability/OC_Attributions_survival', image_normaliser(oc_attributions_grid3), running_iter)
                    break

            with torch.no_grad():
                for images, names, labels in val_loader:
                    images = images.cuda()
                    labels = labels.cuda()
                    labels = labels.float()

                    ## TTA
                    batch_size, n_crops, c, h, w = images.size()
                    images = images.view(-1, c, h, w)
                    out = model(images)
                    out = out.view(batch_size, n_crops, -1).mean(1)

                    # out = model(images)
                    # out = torch.softmax(out, dim=1)
                    val_loss = criterion(out.data, labels)
                    out = torch.softmax(out, dim=1)

                    running_loss += val_loss.item()

                    total += labels.numel()
                    # out = torch.sigmoid(out)

                    # Save validation images for post all folds training aggregation
                    if epoch == (EPOCHS - 1):
                        val_preds += out.cpu().numpy().tolist()
                        val_labels += labels.cpu().numpy().tolist()
                        val_names += names

                    for classID in range(num_classes):
                        acc = ((out[:, classID] > 0.5).int() == labels[:, classID]).sum().item()
                        acc = round(acc, 4)
                        class_correct[classID] += acc
                    # correct += ((out > 0.5).int() == labels).sum().item()

                    res_id += names
                    res_prob += out.cpu().numpy().tolist()
                    res_label += labels.cpu().numpy().tolist()
                    val_counter += 1

            # Write to tensorboard
            writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)

            # acc = correct / total
            class_correct = [i*num_classes/total for i in class_correct]
            y_true = np.array(res_label)
            y_scores = np.array(res_prob)

            # Overalls
            true_auc = roc_auc_score(y_true, y_scores)
            precision_overall, recall_overall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
            true_pr_auc = auc(recall_overall, precision_overall)

            class_auc = []
            for classID in range(num_classes):
                auc_score = roc_auc_score(y_true[:, classID], y_scores[:, classID])
                class_auc.append(auc_score)
            class_pr = []
            for classID in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, classID], y_scores[:, classID])
                pr_auc = auc(recall, precision)
                class_pr.append(pr_auc)

            # Aggregation
            if epoch == (EPOCHS - 1):
                overall_val_roc_aucs.append(true_auc)
                overall_val_pr_aucs.append(true_pr_auc)
                class_val_roc_aucs.append(class_auc)
                class_val_pr_aucs.append(class_pr)
            print("Epoch: {}, Loss: {},\n Test Accuracy: {},\n ROC-AUCs: {},\n PR-AUCs {}\n".format(epoch,
                                                                                                    running_loss,
                                                                                                    class_correct,
                                                                                                    class_auc,
                                                                                                    class_pr))
            writer.add_scalars(f'Loss/AUCs', {
                '48H': class_auc[0],
                '1week-': class_auc[1],
                '1week+': class_auc[2],
                'survival': class_auc[3],
            }, running_iter)
            writer.add_scalars(f'Loss/PR_AUCs', {
                '48H': class_pr[0],
                '1week-': class_pr[1],
                '1week+': class_pr[2],
                'survival': class_pr[3],
            }, running_iter)
            writer.add_scalar('Loss/AUC', true_auc, running_iter)
            writer.add_scalar('Loss/PR_AUC', np.mean(class_pr), running_iter)

            # Save model
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
                print(MODEL_PATH)
                # if epoch != (EPOCHS - 1):
                #     torch.save({'model_state_dict': model.state_dict(),
                #                 'optimizer_state_dict': optimizer.state_dict(),
                #                 'scheduler_state_dict': scheduler.state_dict(),
                #                 'epoch': epoch,
                #                 'loss': loss,
                #                 'running_iter': running_iter,
                #                 'encoder': encoder,
                #                 'batch_size': bs,
                #                 'resolution': input_size}, MODEL_PATH)
                # elif epoch == (EPOCHS - 1):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'loss': loss,
                            'running_iter': running_iter,
                            'encoder': encoder,
                            'batch_size': bs,
                            'resolution': input_size,
                            'val_preds': val_preds,
                            'val_labels': val_labels,
                            'overall_val_roc_aucs': overall_val_roc_aucs,
                            'overall_val_pr_aucs': overall_val_pr_aucs,
                            'class_val_roc_aucs': class_val_roc_aucs,
                            'class_val_pr_aucs': class_val_pr_aucs}, MODEL_PATH)

    # Now that this fold's training has ended, want starting points of next fold to reset
    latest_epoch = -1
    latest_fold = 0
    running_iter = 0

    # Print various fold outputs: Sanity check
    print(f'Fold {fold} val_preds: {val_preds}')
    print(f'Fold {fold} val_labels: {val_labels}')
    print(f'Fold {fold} overall_val_roc_aucs: {overall_val_roc_aucs}')
    print(f'Fold {fold} overall_val_pr_aucs: {overall_val_pr_aucs}')
    print(f'Fold {fold} class_val_roc_aucs: {class_val_roc_aucs}')
    print(f'Fold {fold} class_val_pr_aucs: {class_val_pr_aucs}')


## Totals
val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

overall_val_roc_aucs = np.array(overall_val_roc_aucs)
overall_val_pr_aucs = np.array(overall_val_pr_aucs)

class_val_roc_aucs = np.array(class_val_roc_aucs)
class_val_pr_aucs = np.array(class_val_pr_aucs)

# Folds analysis
print('Labels', len(val_labels), 'Preds', len(val_preds), 'AUCs', len(overall_val_roc_aucs))
correct = ((val_preds > 0.5).astype(int) == val_labels).sum()
acc = correct / len(val_labels)

# Folds AUCs
folds_roc_auc = roc_auc_score(val_labels, val_preds)
precision_folds, recall_folds, _ = precision_recall_curve(val_labels.ravel(), val_preds.ravel())
folds_pr_auc = auc(recall_folds, precision_folds)
print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), folds_roc_auc))
print('ROC AUC mean:', np.mean(overall_val_roc_aucs), 'std:', np.std(overall_val_roc_aucs))
print('PR AUC mean:', np.mean(overall_val_pr_aucs), 'std:', np.std(overall_val_pr_aucs))

res_prob = [x[0] for x in res_prob]
sub = pd.DataFrame({"Filename": val_names, "Died": val_labels.tolist(), "Pred": val_preds.tolist()})
sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

## Plot
# Compute ROC curve and ROC area for each class
class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']

fpr = dict()
tpr = dict()
roc_auc = dict()
for classID in range(num_classes):
    fpr[classID], tpr[classID], _ = roc_curve(val_labels[:, classID], val_preds[:, classID])
    roc_auc[classID] = auc(fpr[classID], tpr[classID])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(val_labels.ravel(), val_preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute PR curve and PR area for each class
precision_tot = dict()
recall_tot = dict()
pr_auc = dict()
for classID in range(num_classes):
    precision_tot[classID], recall_tot[classID], _ = precision_recall_curve(val_labels[:, classID], val_preds[:, classID])
    pr_auc[classID] = auc(recall_tot[classID], precision_tot[classID])

# Compute micro-average precision-recall curve and PR area
precision_tot["micro"], recall_tot["micro"], _ = precision_recall_curve(val_labels.ravel(), val_preds.ravel())
pr_auc["micro"] = auc(recall_tot["micro"], precision_tot["micro"])
no_skill = len(val_labels[val_labels == 1]) / len(val_labels)

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
plt.savefig('roc-' + encoder + '-bs%d-%d.png' % (bs, input_size[0]), dpi=300)

# PR plot
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
fig_name = f'precision-recall-{encoder}-bs{bs}-{input_size[0]}.png'
plt.savefig(os.path.join(SAVE_PATH, fig_name), dpi=300)

# Save relevant data to csv
val_labels = [x[0] for x in val_labels]
val_preds = [x[0] for x in val_preds]
sub = pd.DataFrame({"Filename": val_names, "Died": val_labels, "Pred": val_preds})
sub_name = f'preds-{encoder}-bs{bs}-{input_size[0]}.csv'
sub.to_csv(os.path.join(SAVE_PATH, sub_name), index=False)

print('END')
