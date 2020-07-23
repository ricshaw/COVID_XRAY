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


# Some necessary variables
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
    EPOCHS = 170
    FOLDS = 5
else:
    running_iter = 0
    loaded_epoch = -1
    bs = 64
    EPOCHS = 170
    FOLDS = 5

# Load labels
print(f'The  labels are {labels}')
if len(labels) == 1:
    labels = labels[0]
    df = pd.read_csv(labels)

elif len(labels) > 1:
    df = pd.read_csv(labels[0])
    for extra in range(1, len(labels)):
        extra_df = pd.read_csv(labels[extra])
        df = pd.concat([df, extra_df], ignore_index=True)

# For shape purposes:
temp_bloods = df[df.columns.difference(df.filter(like='ICU').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='date of death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='OHE').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='stratify').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='fold').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Death').columns, sort=False)]
temp_bloods = temp_bloods[temp_bloods.columns.difference(temp_bloods.filter(like='Died').columns, sort=False)]
temp_bloods = temp_bloods.select_dtypes(include=[np.number])

# # Exclude all entries with "Missing" Died stats
# df = df[~df['Died'].isin(['Missing'])]
# df['Died'] = pd.to_numeric(df['Died'])

# Augmentations
print("Died:", df[df.Died == 1].shape[0])
print("Survived:", df[df.Died == 0].shape[0])


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
mvp_features = []


alpha = 0.75
gamma = 2.0
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
    model = Model()
    # alpha = torch.FloatTensor([0.9, 0.8, 0.7, 0.25])[None, ...].cuda()
    # criterion = FocalLoss(logits=True)
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
        val_preds = checkpoint['val_preds']
        val_labels = checkpoint['val_labels']
        val_names = checkpoint['val_names']
        mvp_features = checkpoint['mvp_features']
        overall_val_roc_aucs = checkpoint['overall_val_roc_aucs']
        overall_val_pr_aucs = checkpoint['overall_val_pr_aucs']
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

    train_dataset = ImageDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=8, shuffle=True)

    val_dataset = ImageDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=int(bs/4), num_workers=8)

    print(f'The shape of the labels are: {df.shape}')
    # for colu in df.columns:
    #     print(colu)
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
                names, labels, bloods = sample[0], sample[1], sample[2]

                labels = labels.cuda()
                labels = labels.unsqueeze(1).float()
                bloods = bloods.cuda()
                bloods = bloods.float()

                out = model(bloods)
                # out = torch.softmax(out, dim=1)
                # print(out.shape, out)

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
                    writer.add_image('Visuals/Rounded Output', image_normaliser(rounded_output_grid), running_iter)

                print("iter: {}, Loss: {}".format(running_iter, loss.item()))
                running_iter += 1

            print("Epoch: {}, Loss: {},\n Train Accuracy: {}".format(epoch, running_loss, train_acc/total))
            # if epoch % 2 == 1:
            #     scheduler.step()

            print('Validation step')
            model.eval()
            running_loss = 0
            # correct = 0
            class_correct = [0]
            val_counter = 0
            total = 0
            res_id = []
            res_prob = []
            res_label = []
            if (epoch == (EPOCHS - 1)) or (epoch % 10 == 0):
                occlusion = True
            else:
                occlusion = False
            if occlusion:
                occlusion_count = 0
                for names, labels, bloods in val_loader:
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

                    # Account for tta: Take first image (non-augmented)
                    # Label does not need to be touched because it is obv. the same for this image regardless of tta
                    # Set a baseline
                    baseline_bloods = torch.zeros_like(bloods).cuda().float()

                    # Calculate attribution scores + delta
                    # ig = IntegratedGradients(model)
                    oc = Occlusion(model)
                    # nt = NoiseTunnel(ig)
                    # attributions, delta = nt.attribute(image, nt_type='smoothgrad', stdevs=0.02, n_samples=2,
                    #                                    baselines=baseline, target=0, return_convergence_delta=True)
                    _, target_ID = torch.max(labels, 1)
                    print(target_ID)
                    print(baseline_bloods.shape)
                    # attributions = ig.attribute(image, baseline, target=target_ID, return_convergence_delta=False)
                    blud0 = oc.attribute(bloods, sliding_window_shapes=(1,),
                                         strides=(1,), target=0,
                                         baselines=baseline_bloods)
                    # print('IG + SmoothGrad Attributions:', attributions)
                    # print('Convergence Delta:', delta)

                    # Print
                    for single_feature in range(blud0.shape[0]):
                        mvp_feature = temp_bloods.columns[int(np.argmax(blud0[single_feature, :].cpu()))]
                        print(f'The most valuable feature was {mvp_feature}')
                        if epoch == (EPOCHS - 1):
                            mvp_features.append(mvp_feature)

                    random_index = np.random.randint(labels.size(0))
                    blud0 = blud0[random_index, :]
                    # Change bluds shape to rectangular for ease of visualisation
                    occ_shape = factor_int(blud0.shape[0])
                    print(f'occ shape is {occ_shape}')
                    blud0_grid = torchvision.utils.make_grid(torch.abs(torch.reshape(blud0, occ_shape)))

                    # Write to tensorboard
                    # Bluds
                    if occlusion_count == 0:
                        writer.add_image('Interpretability/Bloods', image_normaliser(blud0_grid), running_iter)
                    occlusion_count += 1

            with torch.no_grad():
                for names, labels, bloods in val_loader:
                    labels = labels.cuda()
                    labels = labels.unsqueeze(1).float()
                    bloods = bloods.cuda()
                    bloods = bloods.float()

                    out = model(bloods)

                    val_loss = sigmoid_focal_loss(out, labels, alpha=alpha, gamma=gamma, reduction="mean")
                    out = torch.sigmoid(out)

                    running_loss += val_loss.item()

                    total += labels.numel()
                    # out = torch.sigmoid(out)

                    # Save validation output for post all folds training aggregation
                    if epoch == (EPOCHS - 1):
                        val_preds += out.cpu().numpy().tolist()
                        val_labels += labels.cpu().numpy().tolist()
                        val_names += names

                    acc = ((out > 0.5).int() == labels).sum().item()
                    # correct += ((out > 0.5).int() == labels).sum().item()

                    res_id += names
                    res_prob += out.cpu().numpy().tolist()
                    res_label += labels.cpu().numpy().tolist()
                    val_counter += 1

            # Write to tensorboard
            writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)

            # acc = correct / total
            acc = ((out > 0.5).int() == labels).sum().item()
            val_acc = acc / total
            y_true = np.array(res_label)
            y_scores = np.array(res_prob)

            # Overalls
            true_auc = roc_auc_score(y_true, y_scores)
            precision_overall, recall_overall, _ = precision_recall_curve(y_true.ravel(), y_scores.ravel())
            true_pr_auc = auc(recall_overall, precision_overall)

            # Aggregation
            if epoch == (EPOCHS - 1):
                overall_val_roc_aucs.append(true_auc)
                overall_val_pr_aucs.append(true_pr_auc)
            print("Epoch: {}, Loss: {},\n Test Accuracy: {},\n ROC-AUCs: {},\n PR-AUCs {}\n".format(epoch,
                                                                                                    running_loss,
                                                                                                    val_acc,
                                                                                                    true_auc,
                                                                                                    true_pr_auc))
            writer.add_scalar('Loss/AUC', true_auc, running_iter)
            writer.add_scalar('Loss/PR_AUC', true_pr_auc, running_iter)

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
                            'val_preds': val_preds,
                            'val_labels': val_labels,
                            'overall_val_roc_aucs': overall_val_roc_aucs,
                            'overall_val_pr_aucs': overall_val_pr_aucs,
                            'mvp_features': mvp_features,
                            'val_names': val_names}, MODEL_PATH)

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
val_labels = np.array(val_labels)
val_preds = np.array(val_preds)

overall_val_roc_aucs = np.array(overall_val_roc_aucs)
overall_val_pr_aucs = np.array(overall_val_pr_aucs)

# Folds analysis
print('Labels', len(val_labels), 'Preds', len(val_preds), 'AUCs', len(overall_val_roc_aucs))
correct = ((val_preds > 0.5).astype(int) == val_labels).sum()
acc = correct / len(val_labels)

# Folds AUCs
folds_roc_auc = roc_auc_score(val_labels, val_preds)
precision_folds, recall_folds, _ = precision_recall_curve(val_labels.ravel(), val_preds.ravel())
folds_pr_auc = auc(recall_folds, precision_folds)
# print("Total Accuracy: {}, AUC: {}".format(round(acc, 4), folds_roc_auc))
print('ROC AUC mean:', np.mean(overall_val_roc_aucs), 'std:', np.std(overall_val_roc_aucs))
print('PR AUC mean:', np.mean(overall_val_pr_aucs), 'std:', np.std(overall_val_pr_aucs))

print(f'Length of val_names, val_labels, val_preds, mvp_features are {len(val_names)},'
      f'{len(val_labels.tolist())}, {len(val_preds.tolist())}, {len(mvp_features)}')
sub = pd.DataFrame({"Filename": val_names, "Died": val_labels.tolist(), "Pred": val_preds.tolist(), "MVP_feat": mvp_features})
sub.to_csv(os.path.join(SAVE_PATH, 'preds.csv'), index=False)

## Plot
# Compute ROC curve and ROC area for each class
class_names = ['48H', '1 week -', '1 week +', 'Survived', 'micro']

fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(val_labels.ravel(), val_preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute PR curve and PR area for each class
precision_tot = dict()
recall_tot = dict()
pr_auc = dict()

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
val_labels = [x[0] for x in val_labels]
val_preds = [x[0] for x in val_preds]
sub = pd.DataFrame({"Filename": val_names, "Died": val_labels, "Pred": val_preds, "MVP_feat": mvp_features})
sub_name = f'preds-bs{bs}-logreg-{arguments.job_name}.csv'
sub.to_csv(os.path.join(SAVE_PATH, sub_name), index=False)

print('END')
