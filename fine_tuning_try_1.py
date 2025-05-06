import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# --- Dataset ---
class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        label = self.df.loc[idx, "label"]
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Model ---
def get_efficientnet(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# --- Training Utilities ---
def set_requires_grad(model, head_only=True, unfreeze_blocks=0):
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze head
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Unfreeze top blocks if needed
    if unfreeze_blocks > 0:
        # EfficientNet blocks: model.features[0] ... model.features[6]
        for i in range(7 - unfreeze_blocks, 7):
            for param in model.features[i].parameters():
                param.requires_grad = True

def get_optimizer(model, lr_head, lr_backbone):
    # Different learning rates for head and backbone
    params = [
        {"params": model.classifier.parameters(), "lr": lr_head},
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad], "lr": lr_backbone},
    ]
    return optim.Adam(params)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), all_preds, all_labels

def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    bin_preds = (np.array(all_preds) > threshold).astype(int)
    f1 = f1_score(all_labels, bin_preds)
    precision = precision_score(all_labels, bin_preds)
    recall = recall_score(all_labels, bin_preds)
    return running_loss / len(loader.dataset), f1, precision, recall, all_preds, all_labels

def optimize_threshold(y_true, y_probs):
    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (np.array(y_probs) > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# --- Test-Time Augmentation ---
def tta_predict(model, img, device, tta_transforms):
    model.eval()
    preds = []
    with torch.no_grad():
        for t in tta_transforms:
            aug_img = t(img)
            aug_img = aug_img.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(aug_img)).cpu().numpy()
            preds.append(pred)
    return np.mean(preds)

# --- Main Training Loop ---
def run_fold(
    fold, train_df, val_df, num_classes, device, writer, tta_transforms, 
    epochs_head=10, epochs_ft=10, unfreeze_blocks=2, patience=5
):
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_set = FERDataset(train_df, train_transform)
    val_set = FERDataset(val_df, val_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = get_efficientnet(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # --- Phase 1: Train head only ---
    set_requires_grad(model, head_only=True)
    optimizer = get_optimizer(model, lr_head=1e-3, lr_backbone=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    best_f1, best_model = 0, None
    patience_counter = 0

    for epoch in range(epochs_head):
        train_loss, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_prec, val_rec, val_probs, val_labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)
        writer.add_scalar(f"Fold{fold}/TrainLoss_head", train_loss, epoch)
        writer.add_scalar(f"Fold{fold}/ValLoss_head", val_loss, epoch)
        writer.add_scalar(f"Fold{fold}/ValF1_head", val_f1, epoch)
        writer.add_scalar(f"Fold{fold}/ValPrecision_head", val_prec, epoch)
        writer.add_scalar(f"Fold{fold}/ValRecall_head", val_rec, epoch)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (head)")
            break

    # --- Phase 2: Progressive unfreezing ---
    model.load_state_dict(best_model)
    set_requires_grad(model, head_only=False, unfreeze_blocks=unfreeze_blocks)
    optimizer = get_optimizer(model, lr_head=1e-4, lr_backbone=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    best_f1, best_model = 0, None
    patience_counter = 0

    for epoch in range(epochs_ft):
        train_loss, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_prec, val_rec, val_probs, val_labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)
        writer.add_scalar(f"Fold{fold}/TrainLoss_ft", train_loss, epoch)
        writer.add_scalar(f"Fold{fold}/ValLoss_ft", val_loss, epoch)
        writer.add_scalar(f"Fold{fold}/ValF1_ft", val_f1, epoch)
        writer.add_scalar(f"Fold{fold}/ValPrecision_ft", val_prec, epoch)
        writer.add_scalar(f"Fold{fold}/ValRecall_ft", val_rec, epoch)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (ft)")
            break

    # --- Threshold optimization ---
    model.load_state_dict(best_model)
    _, _, _, _, val_probs, val_labels = validate(model, val_loader, criterion, device)
    best_thr, best_f1 = optimize_threshold(val_labels, val_probs)
    print(f"Fold {fold}: Best threshold {best_thr:.2f}, F1 {best_f1:.4f}")

    # --- TTA on validation set ---
    tta_preds = []
    for idx in range(len(val_set)):
        img, _ = val_set[idx]
        pred = tta_predict(model, img, device, tta_transforms)
        tta_preds.append(pred)
    tta_bin_preds = (np.array(tta_preds) > best_thr).astype(int)
    tta_f1 = f1_score(val_labels, tta_bin_preds)
    print(f"Fold {fold}: TTA F1 {tta_f1:.4f}")

    return best_f1, tta_f1

# --- Main Cross-Validation ---
def main():
    # Load your dataframe
    df = pd.read_csv("your_data.csv")  # columns: image_path, label
    num_classes = 1  # binary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/efficientnet_smile")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tta_transforms = [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(1.0), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(10), transforms.ToTensor()]),
    ]

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["image_path"], df["label"])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        best_f1, tta_f1 = run_fold(
            fold, train_df, val_df, num_classes, device, writer, tta_transforms
        )
        fold_results.append((best_f1, tta_f1))
        print(f"Fold {fold}: Val F1={best_f1:.4f}, TTA F1={tta_f1:.4f}")

    print("Cross-validation results:")
    for i, (f1, tta_f1) in enumerate(fold_results):
        print(f"Fold {i}: F1={f1:.4f}, TTA F1={tta_f1:.4f}")
    print("Mean F1:", np.mean([f[0] for f in fold_results]))
    print("Mean TTA F1:", np.mean([f[1] for f in fold_results]))

if __name__ == "__main__":
    main()
