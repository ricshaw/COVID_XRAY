import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2
import sys
import random
import pandas as pd
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer, required
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms, models

from efficientnet_pytorch import EfficientNet
import albumentations as A
from focal_loss import sigmoid_focal_loss, sigmoid_focal_loss_star

sys.path.append('/nfs/home/richard/over9000')
from rangerlars import RangerLars

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

USE_HPO = False
USE_JPG = True
USE_TTA = True

if USE_HPO:
    import runai.hpo
    strategy = runai.hpo.Strategy.GridSearch
    runai.hpo.init('/nfs/project/richard', 'covid-primary-bloods')
    config = runai.hpo.pick(
    grid=dict(
            batch_size=[4,8,16,32],
            lr=[0.1,0.01,0.001],
            aug=[0.1,0.2,0.3,0.4,0.5],
            chns1=[32,64,128,256],
            chns2=[32,64,128,256],
            dropout=[0.1,0.2,0.3,0.4,0.5]),
    strategy=strategy)
else:
    #model-kch-bs32-lr0.001-dp0.3-epochs30-efficientnet-b3-sz512
    config = dict(
            epochs=30,
            batch_size=32,
            lr=0.001,
            aug=0.1,
            chns1=32,
            chns2=32,
            dropout=0.3,
            image_sz=512,
            encoder='efficientnet-b3')
    print('Config:', config)

if USE_JPG:
    try:
        __import__('turbojpeg')
    except ImportError:
        os.system('pip install /nfs/home/richard/PyTurboJPEG.zip')
    from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
    jpeg = TurboJPEG()
    def load_jpeg(f):
        in_file = open(f, 'rb')
        bgr_array = jpeg.decode(in_file.read())
        in_file.close()
        return bgr_array

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


ROOT = '/nfs/home/richard/PRIMARY_OBJECTIVE'
TRAIN = True
TEST = True
TEST_SET = 'KCH'
#TEST_SET = 'GSTT'
IMAGES = True
FEATS = False
CENTRE_CROP = True
FOLDS = 5
SAVE = True
SAVE_NAME = 'model-kch-bs%d-lr%.03f-dp%.01f-epochs%d' % (config['batch_size'], config['lr'], config['dropout'], config['epochs'])
if IMAGES:
    SAVE_NAME += '-' + config['encoder'] + '-sz%d' % config['image_sz']
if FEATS:
    SAVE_NAME += '-chns1_%d-chns2_%d-aug%.01f' % (config['chns1'], config['chns2'], config['aug'])
SAVE_PATH = os.path.join(ROOT, SAVE_NAME)
print('Model:', SAVE_NAME)
if SAVE:
    os.makedirs(SAVE_PATH, exist_ok=True)
    log_name = os.path.join(SAVE_PATH, 'run')
    writer = SummaryWriter(log_dir=log_name)

df = pd.read_csv(os.path.join(ROOT,'cxr_folds_filter.csv'))
if TEST_SET=='KCH':
    test_df = pd.read_csv(os.path.join(ROOT,'KCH_folds.csv'))
if TEST_SET=='GSTT':
    test_df = pd.read_csv(os.path.join(ROOT,'GSTT_folds.csv'))
#df = pd.read_csv(os.path.join(ROOT,'GSTT_folds.csv'))
#test_df = pd.read_csv(os.path.join(ROOT,'KCH_folds.csv'))
#train_dir = '/nfs/project/covid/CXR/primary_obj_imgs_kch'
#test_dir = '/nfs/project/covid/CXR/GSTT/primary_obj_imgs'
df.columns = df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()
test_df = test_df.rename(columns={'death': 'died', 'accession number':'accession'})
test_df.filename = [f.split('/')[-1] for f in test_df.filename]

if USE_JPG:
    train_dir = '/nfs/project/covid/CXR/KCH_CXR_JPG'
    if TEST_SET=='KCH':
        test_dir = '/nfs/project/covid/CXR/KCH_CXR_JPG'
    if TEST_SET=='GSTT':
        test_dir = '/nfs/project/covid/CXR/GSTT/primary_obj_imgs_jpg'
else:
    train_dir = '/nfs/project/covid/CXR/KCH_CXR_PNG'
    if TEST_SET=='KCH':
        test_dir = '/nfs/project/covid/CXR/KCH_CXR_PNG'
    if TEST_SET=='GSTT':
        test_dir = '/nfs/project/covid/CXR/GSTT/primary_obj_imgs'
print('Train data:', df.shape)

#BLOOD_COLS = ['Lymphocytes','Albumin','Estimated GFR','PCV','PLT','Creatinine','WBC','C-reactive Protein','Urea',
#              'INR','Sodium','Bilirubin (Total)','.pCO2','FiO2','Heart Rate','Temperature','Oxygen Saturation','Temperature_Max']
#BLOOD_COLS = ['Lymphocytes','Albumin','Estimated GFR','PCV','PLT','Creatinine','WBC','C-reactive Protein','Urea',
#              'INR','Sodium','Bilirubin (Total)','.pCO2','FiO2','Oxygen Saturation']
#BLOOD_COLS = ['PLT','Creatinine','WBC']
#BLOOD_COLS = ['PLT']
#BLOOD_COLS = ['PCV']
BLOOD_COLS = ['urea']
TARGET = 'died'

def prepare_data(in_df, blood_cols, scaler, imputer, fit_scaler=False, fit_imputer=False):
    df = in_df.copy()

    ## Gender
    df['male'] = 0
    df['female'] = 0
    df.loc[df['client_gendercode'] == 1, 'male'] = 1
    df.loc[df['client_gendercode'] == 0, 'female'] = 1

    # Extract features
    bloods = df.loc[:,blood_cols].values.astype(np.float32)
    age = df.age.values[:,None]

    # Normalise features
    X = np.concatenate((bloods, age), axis=1)
    if fit_scaler:
        print('Fitting scaler')
        scaler.fit(X)
    X = scaler.transform(X)

    # Fill missing
    if fit_imputer:
        print('Fitting imputer')
        imputer.fit(X)
    X = imputer.transform(X)

    # Put back features
    df.loc[:,blood_cols] = X[:,0:bloods.shape[1]]
    df.loc[:,'age'] = X[:,bloods.shape[1]]
    return df


def default_image_loader(path):
    img = Image.open(path).convert('RGB')
    return img

def get_image(df, filename, transform, A_transform=None):
    if USE_JPG:
        image = load_jpeg(filename)
        if CENTRE_CROP:
            sz = min(image.shape[:2])
            image = A.augmentations.transforms.CenterCrop(sz, sz, always_apply=True)(image=image)['image']
    else:
        image = default_image_loader(filename)
        if CENTRE_CROP:
            image = transforms.CenterCrop(min(image.size))(image)
    # A transform
    if A_transform is not None:
        image = np.array(image)
        image = A_transform(image=image)['image']

    if type(image.size) is not tuple:
        image = Image.fromarray(image)
    # Transform
    image = transform(image)
    return image

def get_feats(df, i, aug=False):
    male = df.male[i].astype(np.float32)
    female = df.female[i].astype(np.float32)
    age = df.age[i].astype(np.float32)
    white = df.white[i].astype(np.float32)
    black = df.black[i].astype(np.float32)
    asian = df.asian[i].astype(np.float32)
    bloods = df.loc[i, BLOOD_COLS].values.astype(np.float32)
    if aug:
        bloods += np.random.normal(0, config['aug'], bloods.shape)
    #feats = np.concatenate((bloods, [male, female, age, white, black, asian]), axis=0)
    #feats = np.array([male, female, age, white, black, asian])
    feats = np.array([male, female, age])
    return feats


class MyDataset(Dataset):
    def __init__(self, my_df, my_dir, transform, A_transform=None, feats_aug=False):
        self.df = my_df
        self.dir = my_dir
        self.loader = default_image_loader
        self.transform = transform
        self.A_transform = A_transform
        self.feats_aug = feats_aug

    def __getitem__(self, index):
        image, feats = np.array([]), np.array([])
        if USE_JPG:
            filename = os.path.join(self.dir, self.df.filename[index][:-3] + 'jpg')
        else:
            filename = os.path.join(self.dir, self.df.filename[index][:-3] + 'png')
        if IMAGES:
            image = get_image(self.df, filename, self.transform, self.A_transform)
        if FEATS:
            feats = get_feats(self.df, index, self.feats_aug)
        label = self.df[TARGET][index]
        name = self.df['accession'][index]
        return name, image, feats, label

    def __len__(self):
        return self.df.shape[0]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
val_transform = transforms.Compose([
                              transforms.Resize((config['image_sz'],config['image_sz']), 3),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)
                            ])
A_transform = A.Compose([
                         A.Resize(config['image_sz'], config['image_sz'], interpolation=2, p=1),
                         A.Flip(p=0.5),
                         A.RandomRotate90(p=1),
                         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, interpolation=2, border_mode=0, p=0.5),
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
                         A.OneOf([
                                  A.OpticalDistortion(interpolation=3, p=0.1),
                                  A.GridDistortion(interpolation=3, p=0.1),
                                  A.IAAPiecewiseAffine(p=0.5),
                                 ], p=0.2),
                         A.OneOf([
                                  A.CLAHE(clip_limit=2),
                                  A.IAASharpen(),
                                  A.IAAEmboss(),
                                 ], p=0.2),
                         A.RandomBrightnessContrast(p=0.5),
                         A.RandomGamma(p=0.5),
                         #A.ToGray(p=1),
                         A.InvertImg(p=0.1),
                         A.CoarseDropout(max_holes=16, max_height=int(0.1*config['image_sz']), max_width=int(0.1*config['image_sz']), fill_value=0, p=0.5),
                        ], p=1)
tta_transform = transforms.Compose([transforms.Resize((config['image_sz'],config['image_sz']), 3),
                                    transforms.Lambda(lambda image: torch.stack([
transforms.ToTensor()(image),
transforms.ToTensor()(image.rotate(90, resample=0)),
transforms.ToTensor()(image.rotate(180, resample=0)),
transforms.ToTensor()(image.rotate(270, resample=0)),
transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM)),
transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(90, resample=0)),
transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(180, resample=0)),
transforms.ToTensor()(image.transpose(method=Image.FLIP_TOP_BOTTOM).rotate(270, resample=0)),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
#transforms.ToTensor()(Image.fromarray(A_transform(image=np.array(image))['image'])),
                                                                     ])),
                                         transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean, std)(image) for image in images]))
                                       ])



class Model(nn.Module):
    def __init__(self, encoder='efficientnet-b0', nfeats=24, mode='train'):
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
        if IMAGES:
            if mode=='train':
                self.net = EfficientNet.from_pretrained(encoder)
            if mode=='test':
                self.net = EfficientNet.from_name(encoder)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.out_chns += n_channels_dict[encoder]
        if FEATS:
            hidden1 = config['chns1']
            hidden2 = config['chns2']
            self.out_chns += hidden2
            self.fc1 = nn.Linear(nfeats, hidden1, bias=False)
            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.meta = nn.Sequential(self.fc1,
                                      nn.ReLU(),
                                      nn.Dropout(config['dropout']),
                                      self.fc2,
                                      nn.ReLU(),
                                      nn.Dropout(config['dropout']),)
        self.fc3 = nn.Linear(self.out_chns, 1)

    def forward(self, image=None, feats=None):
        x1 = torch.FloatTensor().cuda()
        x2 = torch.FloatTensor().cuda()
        if IMAGES:
            x1 = self.net.extract_features(image)
            x1 = self.avg_pool(x1)
            x1 = nn.Flatten()(x1)
        if FEATS:
            x2 = self.meta(feats)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        return x


## Check train dataloader
if False:
    check_dataset = MyDataset(df, train_transform, A_transform=None, feats_aug=False)
    check_loader = DataLoader(check_dataset, batch_size=1, num_workers=0, shuffle=False)
    for i, sample in enumerate(check_loader):
        name, image, feats, label = sample[0], sample[1], sample[2], sample[3]
        image = image.cpu().numpy()[0]
        feats = feats.cpu().numpy()[0]
        print(i, image.shape, feats.shape, label.item())
    exit(0)


def train_epoch(model, optimizer, loader, alpha=0.75):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    res_name, res_prob, res_label = [], [], []
    for i, sample in enumerate(loader):
        name, image, feats, label = sample[0], sample[1], sample[2], sample[3]
        image, feats, label = image.cuda(), feats.cuda(), label.cuda()
        label = label.unsqueeze(1).float()
        out = model(image, feats)
        loss = sigmoid_focal_loss(out, label, alpha, gamma=2.0, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total += label.size(0)
        out = torch.sigmoid(out)
        correct += ((out > 0.5).int() == label).sum().item()
        res_prob += out.detach().cpu().numpy().tolist()
        res_label += label.detach().cpu().numpy().tolist()
    y_true = np.array(res_label)
    y_pred = np.array(res_prob)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
    print("Train Loss: {}, Accuracy: {}, AUC: {}".format(round(epoch_loss,4), round(acc, 4), round(auc, 4)))
    return epoch_loss, image

def val_epoch(model, loader, alpha=0.75):
    model.eval()
    with torch.no_grad():
        epoch_loss, correct, total = 0, 0, 0
        y_name, y_pred, y_true = [], [], []
        for i, sample in enumerate(loader):
            name, image, feats, label = sample[0], sample[1].cuda(), sample[2].cuda(), sample[3].cuda()
            label = label.unsqueeze(1).float()
            if USE_TTA:
                batch_size, n_crops, c, h, w = image.size()
                image = image.view(-1, c, h, w)
                if FEATS:
                    _, n_feats = feats.size()
                    feats = feats.repeat(1,n_crops).view(-1,n_feats)
                out = model(image, feats)
                out = out.view(batch_size, n_crops, -1).mean(1)
            else:
                out = model(image, feats)
            loss = sigmoid_focal_loss(out, label, alpha, gamma=2.0, reduction="mean")
            epoch_loss += loss.item()
            total += label.size(0)
            out = torch.sigmoid(out)
            correct += ((out > 0.5).int() == label).sum().item()
            y_pred += out.detach().cpu().numpy().tolist()
            y_true += label.detach().cpu().numpy().tolist()
            y_name += name
        y_pred = np.array([x[0] for x in y_pred])
        y_true = np.array([x[0] for x in y_true])
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        print("Val Loss: {}, Accuracy: {}, AUC: {}".format(round(epoch_loss,4), round(acc, 4), round(auc, 4)))
    return y_pred, y_true, y_name, auc, acc, image


def run_fold(fold, df):
    print('\nFold:', fold)

    ## Prepare data
    train_df = df[df['fold']!=fold].reset_index(drop=True).copy()
    val_df = df[df['fold']==fold].reset_index(drop=True).copy()

    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')

    train_df = prepare_data(train_df, BLOOD_COLS, scaler, imputer, fit_scaler=True, fit_imputer=True).reset_index(drop=True, inplace=False)
    val_df = prepare_data(val_df, BLOOD_COLS, scaler, imputer, fit_scaler=False, fit_imputer=False).reset_index(drop=True, inplace=False)
    print('Train:', train_df.shape)
    print('Val:', val_df.shape)

    ## Train dataset
    train_dataset = MyDataset(train_df, train_dir, train_transform, A_transform=A_transform, feats_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True, drop_last=False)

    ## Val dataset
    if USE_TTA:
        val_dataset = MyDataset(val_df, train_dir, tta_transform, A_transform=None, feats_aug=False)
    else:
        val_dataset = MyDataset(val_df, train_dir, val_transform, A_transform=None, feats_aug=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=False)

    ## Init model
    model = Model(config['encoder'], mode='train').cuda()
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters(), lr=config['lr'])
    alpha = train_df[train_df[TARGET]==0].shape[0]/train_df.shape[0]
    print('Alpha:', alpha)

    running_acc, running_auc, running_preds = [], [], []
    smooth_accs, smooth_aucs = [], []
    best_auc, best_acc = 0, 0
    stop_count = 0

    for epoch in range(config['epochs']):
        print('\nEpoch:', epoch)
        ## Train step
        train_loss, train_images = train_epoch(model, optimizer, train_loader, alpha)
        if IMAGES:
            # Tensorboard
            grid = torchvision.utils.make_grid(train_images, nrow=4, normalize=True, scale_each=True)
            writer.add_image('train/images', grid, epoch)

        ## Val step
        y_pred, y_true, y_name, auc, acc, val_images = val_epoch(model, val_loader, alpha)
        running_auc.append(auc)
        running_acc.append(acc)
        running_preds.append(y_pred)
        smooth_auc = np.mean(running_auc[-3:])
        smooth_acc = np.mean(running_acc[-3:])
        smooth_aucs.append(smooth_auc)
        smooth_accs.append(smooth_acc)
        id = np.argmax(smooth_aucs)
        print("Smooth Val Acc: {}, AUC: {}".format(round(smooth_acc, 4), round(smooth_auc, 4)))
        print('Best Result -- Epoch:', id, 'Acc:', round(smooth_accs[id],4), 'AUC:', round(smooth_aucs[id],4))
        if IMAGES:
            # Tensorboard
            grid = torchvision.utils.make_grid(val_images, nrow=4, normalize=True, scale_each=True)
            writer.add_image('val/images', grid, epoch)

        # Save best model so far
        if smooth_auc >= best_auc:
            best_auc = smooth_auc
            stop_count = 0
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, ('best_fold_%d.pth' % (fold)))
                print('Saving', MODEL_PATH)
                torch.save(model.state_dict(), MODEL_PATH)
        else:
            stop_count += 1

        # Stopping
        if epoch==(config['epochs']-1):
            print('Stopping!')
            y_pred = running_preds[id]
            acc = running_acc[id]
            auc = running_auc[id]
            del model
            torch.cuda.empty_cache()
            print('Model deleted')
            return pd.DataFrame({'name':y_name, 'label':y_true.astype(int), 'pred':y_pred, 'fold':fold, 'auc':auc, 'acc': acc}), \
                   scaler, imputer


def run_test_fold(fold, test_df, scaler, imputer):
    print('\nFold %d' % fold)
    test_df = prepare_data(test_df, BLOOD_COLS, scaler, imputer, fit_scaler=True, fit_imputer=True).reset_index(drop=True, inplace=False)
    if USE_TTA:
        test_dataset = MyDataset(test_df, test_dir, tta_transform, A_transform=None, feats_aug=False)
    else:
        test_dataset = MyDataset(test_df, test_dir, val_transform, A_transform=None, feats_aug=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=4)

    ## Load best model!
    model = Model(config['encoder'], mode='test').cuda()
    model = nn.DataParallel(model)
    MODEL_PATH = os.path.join(SAVE_PATH, ('best_fold_%d.pth' % (fold)))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print('Loaded:', MODEL_PATH)
    else:
        print('No model!')

    model.cuda()
    y_pred, y_true, y_name, auc, acc, val_images = val_epoch(model, test_loader)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame({'name':y_name, 'label':y_true.astype(int), ('fold %d' % fold):y_pred}), auc, acc



def main():
    ## Training
    if TRAIN:
        out_df = pd.DataFrame()
        scalers = []
        imputers = []

        for fold in range(FOLDS):
            fold_df, scaler, imputer = run_fold(fold, df)
            print(fold_df.head(200))
            out_df = out_df.append(fold_df).reset_index(drop=True)
            scalers.append(scaler)
            imputers.append(imputer)
        print(out_df.head(200))

        y_true = out_df['label']
        y_pred = out_df['pred']
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred)
        print("Total Accuracy: {}, AUC: {}".format(round(acc,4), round(auc,4)))
        print('Accuracy mean:', round(np.mean(out_df['acc']),4), 'std:', round(np.std(out_df['acc']),4))
        print('AUC mean:', round(np.mean(out_df['auc']),4), 'std:', round(np.std(out_df['auc']),4))
        out_df.to_csv(os.path.join(SAVE_PATH,'preds-KCH-' + SAVE_NAME + '.csv'), index=False)

    ## Testing
    if TEST:
        print('Testing!')
        print('Model:', SAVE_NAME)
        print('Test data:', test_df.shape)

        out_df = pd.DataFrame()
        y_pred = 0
        test_aucs, test_accs = [], []

        for fold in range(FOLDS):
            #scaler = scalers[fold]
            #imputer = imputers[fold]
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')

            fold_df, auc, acc = run_test_fold(fold, test_df.copy(), scaler, imputer)
            out_df = pd.concat([out_df, fold_df], axis=1).T.drop_duplicates().T
            y_pred += fold_df['fold %d' % fold].values
            test_aucs.append(auc)
            test_accs.append(acc)

        y_pred /= FOLDS
        out_df['pred'] = y_pred
        print(out_df.head(100))

        # Test scores
        y_pred = out_df['pred'].values.astype(np.float32)
        y_true = out_df['label'].values.astype(int)
        acc = balanced_accuracy_score(y_true, (y_pred>0.5).astype(int))
        auc = roc_auc_score(y_true, y_pred)
        print('\nOverall Accuracy:', round(acc,4), 'AUC:', round(auc,4))
        print('Accuracy mean:', round(np.mean(test_accs),4), 'std:', round(np.std(test_accs),4))
        print('AUC mean:', round(np.mean(test_aucs),4), 'std:', round(np.std(test_aucs),4))
        out_df['auc'] = auc
        out_df['acc'] = acc
        out_df.to_csv(os.path.join(SAVE_PATH,'preds-primary-' + TEST_SET + '-' + SAVE_NAME + '.csv'), index=False)


    if USE_HPO:
        ## Report
        val_acc_mean = np.asscalar(np.mean(val_acc))
        val_acc_std = np.asscalar(np.std(val_acc))
        val_auc_mean = np.asscalar(np.mean(val_auc))
        val_auc_std = np.asscalar(np.std(val_auc))

        test_acc_mean = np.asscalar(np.mean(test_accs))
        test_acc_std = np.asscalar(np.std(test_accs))
        test_auc_mean = np.asscalar(np.mean(test_aucs))
        test_auc_std = np.asscalar(np.std(test_aucs))
        runai.hpo.report(epoch=EPOCHS, metrics={'val_acc':val_acc_mean, 'val_acc_std':val_acc_std,
                                        'val_auc':val_auc_mean, 'val_auc_std':val_auc_std,
                                        'test_acc':test_acc_mean, 'test_acc_std':test_acc_std,
                                        'test_auc':test_auc_mean, 'test_auc_std':test_auc_std,
                                        'model':SAVE_NAME })



if __name__ == "__main__":
   main()
