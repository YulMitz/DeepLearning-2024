#!/usr/bin/env python
# coding: utf-8

# ## Import dependencies

# In[11]:


import numpy as np
import pandas as pd
import os
import gc
import sys
import re
import matplotlib.pyplot as plt
import pydicom
import cv2
import glob
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import albumentations as A
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import KFold
from transformers import get_cosine_schedule_with_warmup
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from pathlib import Path

IMAGE_SIZE = 512
INPUT_CHANNELS = 30

N_LABELS = 25
N_CLASSES = 3 * N_LABELS

N_FOLDS = 5

GRAD_ACC = 2
TARGET_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = TARGET_BATCH_SIZE // GRAD_ACC
EVAL_BATCH_SIZE = 8

NUM_EPOCHS = 10
START_EPOCH = 0
EARLY_STOP_EPOCH = 5
SAVE_MODEL_EPOCHS = 4

LEARNING_RATE = 2e-5

BASE_FOLDER = './train_images'  # path to the training dataset (modify it according to your setting)
OUTPUT_DIR = './output'  # path to the output directory (modify it according to your setting)
DEVICE = "cuda"
NUM_WORKERS = int(os.cpu_count() / 2)
SEED = 920  # random seed


# In[3]:


def set_random_seed(seed: int = 920, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

set_random_seed(SEED)


# ## Load & Read the data

# In[4]:


label_coordinates_df = pd.read_csv('./train_label_coordinates.csv')
train_series = pd.read_csv('./train_series_descriptions.csv')
df_train = pd.read_csv('./train.csv')
df_sub = pd.read_csv('./sample_submission.csv')
test_series = pd.read_csv('./test_series_descriptions.csv')

LABELS = list(df_sub.columns[1:])
CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]

LEVELS = [
    'l1_l2',
    'l2_l3',
    'l3_l4',
    'l4_l5',
    'l5_s1',
]

# ## Making Dataset

# In[5]:

# Convert all digit string into int
def atoi(text):
    return int(text) if text.isdigit() else text

# Generate keys(study_id, series_id) for given path
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

study_ids = train_series['study_id'].unique()
sequence_type = list(train_series['series_description'].unique())
bf = BASE_FOLDER


# #### Export png from dcm pictures

# In[6]:


def dcm_to_png(src_path, output_path):
    dcm_image = pydicom.dcmread(src_path)
    image = dcm_image.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255 # Normalize to range(0, 255)

    # Resize and reshape image
    img = cv2.resize(np.array(image), (512, 512), interpolation=cv2.INTER_CUBIC)
    assert img.shape==(512, 512)
    cv2.imwrite(output_path, img)

# ## Training Dataset

# In[7]:


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    print(torch.version.cuda)
    print(torch.__version__)
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Replace na values
df_train = df_train.fillna(-100)

# Transform label to label_int (ex. Normal/Mild -> 0)
label_to_value = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df_train = df_train.replace(label_to_value)

# Define dataset and data loader
class TrainDataset(Dataset):
    def __init__(self, df_train, phase='training', transform=None):
        self.df = df_train
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        images = np.zeros((IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS), dtype=np.uint8)
        row = self.df.iloc[idx]
        study_id = int(row['study_id'])
        cases = [2492114990, 2780132468, 3008676218]
        ignore_case = True if study_id in cases else False
        label = row[1:].values.astype(np.int64) # Label value for 25 classes (5 conditions * 5 levels)
        
        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                png = f'./cvt_png/{study_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(png).convert('L')
                img = np.array(img)
                images[..., i] = img.astype(np.uint8)
            except:
                if not ignore_case:
                    print(f'failed to load on {study_id}, Sagittal T1')
                pass

        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                png = f'./cvt_png/{study_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(png).convert('L')
                img = np.array(img)
                images[..., i + 10] = img.astype(np.uint8)
            except:
                if not ignore_case:
                    print(f'failed to load on {study_id}, Sagittal T2_STIR')
                pass

        # Axial T2
        axt2 = glob.glob(f'./cvt_png/{study_id}/Axial T2/*.png')
        axt2 = sorted(axt2, key=natural_keys)

        step = len(axt2) / 10.0
        start = len(axt2) / 2.0 - 4.0 * step
        end = len(axt2) + 0.0001

        for j, i in enumerate(np.arange(start, end, step)):
            try:
                png_path = axt2[max(0, int((i - 0.5001).round()))]
                img = Image.open(png_path).convert('L')
                img = np.array(img)
                images[..., j + 20] = img.astype(np.uint8)
            except:
                if not ignore_case:
                    print(f'failed to load on {study_id}, Axial T2')
                pass  
            
        assert np.sum(images) > 0

        if self.transform is not None:
            images = self.transform(image = images)['image']
        
        images = images.transpose(2, 0, 1) # Let input channels be the first axis

        return images, label


# ## Transform for Training and Validation

# In[8]:

AUG = True
AUG_PROB = 0.75

transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=0.5, std=0.5)
])


# ## Define Model

# In[9]:


# Define the DenseNet block with batch normalization and ReLU
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return torch.cat((x, out), 1)

# Define the transition layer with batch normalization and average pooling
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.avgpool(out)
        return out

# Define the DenseNet model
class DenseNet(nn.Module):
    def __init__(self, num_classes=75, input_channels=30, growth_rate=32, num_blocks=[6, 12, 24, 16], theta=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Blocks
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(num_blocks):
            block = self._make_dense_block(in_channels, num_layers)
            setattr(self, f'denseblock{i + 1}', block)
            in_channels += num_layers * growth_rate
            if i < len(num_blocks) - 1:
                transition = self._make_transition_layer(in_channels, int(in_channels * theta))
                setattr(self, f'transition{i + 1}', transition)
                in_channels = int(in_channels * theta)

        # Final layers
        self.bn_final = nn.BatchNorm2d(in_channels)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(DenseBlock(in_channels + i * self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels, out_channels):
        return TransitionLayer(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        for i in range(4):
            x = getattr(self, f'denseblock{i + 1}')(x)
            if i < 3:
                x = getattr(self, f'transition{i + 1}')(x)
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# In[10]:


# Setups before training loop
device = torch.device("cuda")
autocast = torch.amp.autocast('cuda', dtype=torch.half)
scaler = torch.amp.GradScaler('cuda', init_scale=4096)
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# In[12]:

def main(df):
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print('#'*30)
        print(f'start fold{fold}')
        print('#'*30)
        print(len(train_idx), len(val_idx))
        df_train = df.iloc[train_idx]
        df_valid = df.iloc[val_idx]

        train_dataset = TrainDataset(df_train=df_train, phase='train', transform=transforms_train)
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=NUM_WORKERS
                )
        
        valid_dataset = TrainDataset(df_valid, phase='valid', transform=transforms_val)
        valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=TRAIN_BATCH_SIZE * 2,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=NUM_WORKERS
                )
        
        model = DenseNet(num_classes=N_CLASSES, input_channels=INPUT_CHANNELS).to(device)

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # model = model.cuda()

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

        warmup_steps = NUM_EPOCHS/10 * len(train_dataloader) // GRAD_ACC
        num_total_steps = NUM_EPOCHS * len(train_dataloader) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_total_steps,
                                                        num_cycles=num_cycles)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights) # Prevent device runtime error

        best_loss = 1.2
        best_wll = 1.2 # Weighted Log Loss
        early_stop_step = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dataloader, leave=True, desc=f'Epoch:{epoch}/{NUM_EPOCHS} in fold {fold}') as pbar:
                optimizer.zero_grad()
                for idx, (imgs ,labels) in enumerate(pbar):
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    with autocast:
                        loss = 0
                        y = model(imgs)
                        for col in range(N_LABELS):
                            pred = y[:, col*3:col*3 + 3] # Severity prediction for each label
                            label = labels[:, col]
                            loss = loss + criterion(pred, label) / N_LABELS

                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)

                    pbar.set_postfix(
                        OrderedDict(
                                loss=f'{loss.item()*GRAD_ACC:.6f}',
                                lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()
            
            train_loss = total_loss/len(train_dataloader)
            print(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            val_preds = [] # Store concat predictions for wll calculation
            val_labels = []

            model.eval()
            with tqdm(valid_dataloader, leave=True, desc=f'(Eval) Epoch:{epoch}/{NUM_EPOCHS} in fold {fold}') as pbar:
                with torch.no_grad():
                    for idx, (imgs, labels) in enumerate(pbar):
                        imgs = imgs.to(device)
                        labels = labels.to(device)

                        with autocast:
                            loss = 0
                            y = model(imgs)
                            for col in range(N_LABELS):
                                pred = y[:, col*3:col*3 + 3]
                                label = labels[:, col]

                                loss = loss + criterion(pred, label) / N_LABELS
                                pred_ = pred.float()
                                val_preds.append(pred_.cpu())
                                val_labels.append(label.cpu())
                            
                            total_loss += loss.item()

            val_loss = total_loss/len(valid_dataloader)

            val_preds = torch.cat(val_preds, dim=0)
            val_labels = torch.cat(val_labels)
            val_wll = criterion2(val_preds, val_labels) # To be compared with val_loss

            pbar.set_description(f'Epoch {epoch + 1}/{NUM_EPOCHS} - val_loss:{val_loss:.6f}, val_wll:{val_wll:.6f}')

            # Determine if model made enough progress
            if val_loss < best_loss or val_wll < best_wll:
                
                early_stop_step = 0

                if device!='cuda:0':
                    model.to('cuda:0')                
                    
                if val_loss < best_loss:
                    print(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss
                    
                if val_wll < best_wll:
                    print(f'epoch:{epoch}, best wll_metric updated from {best_wll:.6f} to {val_wll:.6f}')
                    best_wll = val_wll
                    fname = f'{OUTPUT_DIR}/best_wll_model_norm_fold-{fold}.pt'
                    torch.save(model.state_dict(), fname)
                
                if device!='cuda:0':
                    model.to(device)
                
            else:
                early_stop_step += 1
                if early_stop_step >= EARLY_STOP_EPOCH:
                    print('early stopping')
                    break
            
            if (epoch + 1) % SAVE_MODEL_EPOCHS == 0 or epoch == NUM_EPOCHS - 1:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, Path(OUTPUT_DIR,
                                            f"DenseNetNorm_{fold}_e{epoch}.pth"))


if __name__ == "__main__":
    main(df=df_train)

