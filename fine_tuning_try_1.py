# Cell 1: Imports & Setup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import random
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Cell 2: Data Loading & Preprocessing
df = pd.read_csv('fer2013.csv')
df['label'] = (df['emotion'] == 3).astype(int)  # 3 = happy

def parse_image(pixels):
    arr = np.fromstring(pixels, sep=' ', dtype=np.uint8)
    return arr.reshape(48, 48)

df['image'] = df['pixels'].apply(parse_image)

train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
val_df = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)
test_df = df[df['Usage'] == 'PrivateTest'].reset_index(drop=True)

# Cell 3: Augmentation & Dataset
IMG_SIZE = 224

def get_train_aug():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.7),
        ToTensorV2()
    ])

def get_val_aug():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2()
    ])

def replicate_channels(img):
    return np.stack([img]*3, axis=-1)

class FERDataset(Dataset):
    def __init__(self, df, augment=None):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.df.loc[idx, 'image']
        label = self.df.loc[idx, 'label']
        img = replicate_channels(img)
        if self.augment:
            img = self.augment(image=img)['image']
        else:
            img = torch.tensor(img).permute(2,0,1).float() / 255.
        return img, torch.tensor(label, dtype=torch.float32)

def mixup_cutmix_batch(batch, alpha_mixup=0.2, alpha_cutmix=1.0, p_mixup=0.3, p_cutmix=0.3):
    imgs, labels = batch
    bs = imgs.size(0)
    idx = torch.randperm(bs)
    imgs2, labels2 = imgs[idx], labels[idx]
    r = np.random.rand()
    if r < p_mixup:
        lam = np.random.beta(alpha_mixup, alpha_mixup)
        imgs = lam * imgs + (1 - lam) * imgs2
        labels = lam * labels + (1 - lam) * labels2
    elif r < p_mixup + p_cutmix:
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        W, H = imgs.size(2), imgs.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
        imgs[:, :, y1:y2, x1:x2] = imgs2[:, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        labels = lam * labels + (1 - lam) * labels2
    return imgs, labels

# Cell 4: Model Definition
class EfficientNetHead(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.base = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.base.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.base.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)

# Cell 5: Training Utilities
def train_one_epoch(model, loader, optimizer, criterion, device, mixup_cutmix=False):
    model.train()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if mixup_cutmix:
            imgs, labels = mixup_cutmix_batch((imgs, labels))
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, loader, device, threshold=0.5):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().numpy()
            preds.extend(outputs)
            labels.extend(lbls.numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    bin_preds = (preds > threshold).astype(int)
    f1 = f1_score(labels, bin_preds)
    prec = precision_score(labels, bin_preds)
    rec = recall_score(labels, bin_preds)
    return f1, prec, rec, preds, labels

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_f1 = 0
        self.wait = 0
        self.stop = False

    def __call__(self, f1):
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stop = True

# Cell 6: K-Fold Cross-Validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_FOLDS = 5
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)
    train_ds = FERDataset(train_fold, augment=get_train_aug())
    val_ds = FERDataset(val_fold, augment=get_val_aug())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = EfficientNetHead(dropout=0.5).to(device)
    # Phase 1: Head warm-up
    for param in model.base.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    early_stopping = EarlyStopping(patience=5)
    best_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS_PHASE1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_cutmix=True)
        f1, prec, rec, _, _ = eval_model(model, val_loader, device)
        scheduler.step()
        print(f"Phase 1 | Epoch {epoch}: loss={loss:.4f}, val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
        early_stopping(f1)
        if early_stopping.stop:
            print("Early stopping triggered (Phase 1).")
            break
    model.load_state_dict(best_model_state)

    # Phase 2: Progressive unfreezing
    for name, param in model.base.named_parameters():
        if 'features.6' in name or 'features.7' in name:
            param.requires_grad = True
    params = [
        {'params': [p for n, p in model.named_parameters() if 'fc' in n or 'drop' in n], 'lr': 1e-3},
        {'params': [p for n, p in model.base.named_parameters() if 'features.7' in n], 'lr': 1e-4},
        {'params': [p for n, p in model.base.named_parameters() if 'features.6' in n], 'lr': 1e-5},
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=5)
    best_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS_PHASE2):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_cutmix=True)
        f1, prec, rec, preds, labels = eval_model(model, val_loader, device)
        scheduler.step()
        print(f"Phase 2 | Epoch {epoch}: loss={loss:.4f}, val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            best_preds = preds
            best_labels = labels
        early_stopping(f1)
        if early_stopping.stop:
            print("Early stopping triggered (Phase 2).")
            break
    model.load_state_dict(best_model_state)

    # Threshold optimization
    precs, recs, thresholds = precision_recall_curve(best_labels, best_preds)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    print(f"Fold {fold+1} optimal threshold: {best_thresh:.3f}, F1: {f1s[best_idx]:.4f}")

    # Final metrics at best threshold
    bin_preds = (best_preds > best_thresh).astype(int)
    f1 = f1_score(best_labels, bin_preds)
    prec = precision_score(best_labels, bin_preds)
    rec = recall_score(best_labels, bin_preds)
    print(f"Fold {fold+1} final F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/efficientnet_fold{fold+1}.pth')

    fold_metrics.append({
        'fold': fold+1,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'threshold': best_thresh
    })

# Cell 7: Results Summary
print("\n=== K-Fold Results ===")
for m in fold_metrics:
    print(f"Fold {m['fold']}: F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, Threshold={m['threshold']:.3f}")
print("Average F1: {:.4f}".format(np.mean([m['f1'] for m in fold_metrics])))
print("Average Precision: {:.4f}".format(np.mean([m['precision'] for m in fold_metrics])))
print("Average Recall: {:.4f}".format(np.mean([m['recall'] for m in fold_metrics])))
