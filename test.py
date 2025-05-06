# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from PIL import Image
import os
import random
import copy
import time

# --- Reproducibility ---
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# --- Load Datasets into Dataframe ---
# Path to your FER2013 CSV file
CSV_PATH = "fer2013.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Only use "Training" and "PublicTest" for simplicity
df = df[df["Usage"].isin(["Training", "PublicTest"])]

# Map emotion: 3 (happy) -> 1, others -> 0
df["label"] = (df["emotion"] == 3).astype(int)

# Undersample "not happy" to match "happy"
happy = df[df["label"] == 1]
not_happy = df[df["label"] == 0].sample(len(happy), random_state=SEED)
df_balanced = (
    pd.concat([happy, not_happy])
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print(f"Happy: {len(happy)}, Not Happy (sampled): {len(not_happy)}")
print(df_balanced["label"].value_counts())


class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Handle potential string formatting issues if necessary
        try:
            pixels = np.fromstring(
                row["pixels"], sep=" ", dtype=np.uint8
            ).reshape(48, 48)
        except Exception as e:
            print(f"Error processing row {idx}: {row['pixels'][:50]}...")
            raise e
        img = Image.fromarray(pixels).convert("RGB")
        label = row["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# Transforms for EfficientNetB0 (224x224)
# Added Data Augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10), # Added rotation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Minimal transforms for validation/testing
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- Model Prep ---
def get_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT) # Use updated weights API
    num_ftrs = model.classifier[1].in_features
    # Add Dropout before the final layer as per methodology
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, 2), # Binary classification
    )
    return model

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 16
EPOCHS_PHASE1 = 3 # Epochs for training the head only
EPOCHS_PHASE2 = 7 # Epochs for fine-tuning the whole model
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2
K_FOLDS = 5
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-2
PATIENCE = 5 # Early stopping patience

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
X = df_balanced.index.values # Use index for splitting
y = df_balanced["label"].values

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    train_df = df_balanced.iloc[train_idx]
    val_df = df_balanced.iloc[val_idx]

    # Apply respective transforms
    train_ds = FERDataset(train_df, transform=train_transform)
    val_ds = FERDataset(val_df, transform=val_transform)

    # Use num_workers=0 on Windows or if issues arise
    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Phase 1: Train the Head ---
    print("\n--- Phase 1: Training Head ---")
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Optimizer for head only
    optimizer_head = optim.AdamW(
        model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS_PHASE1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_head.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_head.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - start_time
        print(
            f"Phase 1 - Epoch {epoch+1}/{EPOCHS_PHASE1}, Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
        )

    # --- Phase 2: Fine-tune the whole model ---
    print("\n--- Phase 2: Fine-tuning Full Model ---")
    # Unfreeze backbone
    for param in model.features.parameters():
        param.requires_grad = True

    # Optimizer with differential learning rates
    optimizer_full = optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": LEARNING_RATE_BACKBONE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": LEARNING_RATE_HEAD, # Keep head LR higher
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Early Stopping variables
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS_PHASE2):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_full.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_full.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation step after each epoch
        model.eval()
        val_preds, val_targets = [], []
        val_probs = []
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probabilities.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="binary") # Binary F1
        val_auc = roc_auc_score(val_targets, val_probs)
        epoch_time = time.time() - start_time

        print(
            f"Phase 2 - Epoch {epoch+1}/{EPOCHS_PHASE2}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping Check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best F1: {best_val_f1:.4f}. Saving model state.")
        else:
            patience_counter += 1
            print(f"  -> F1 did not improve. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("  -> Early stopping triggered.")
            break

    # Load best model state for final evaluation for this fold
    if best_model_state:
        print("Loading best model state for final fold evaluation.")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved (perhaps training was too short or F1 never improved). Using last state.")


    # Final Validation for the fold using the best model
    model.eval()
    final_preds, final_targets = [], []
    final_probs = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            final_preds.extend(predicted.cpu().numpy())
            final_targets.extend(labels.numpy())
            final_probs.extend(probabilities.cpu().numpy())

    final_acc = accuracy_score(final_targets, final_preds)
    final_f1 = f1_score(final_targets, final_preds, average="binary")
    final_auc = roc_auc_score(final_targets, final_probs)
    report_dict = classification_report(
        final_targets, final_preds, target_names=["Not Happy", "Happy"], output_dict=True
    )
    report_str = classification_report(
        final_targets, final_preds, target_names=["Not Happy", "Happy"]
    )
    cm = confusion_matrix(final_targets, final_preds)

    print(f"\nFold {fold+1} Final Validation Results:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"AUC: {final_auc:.4f}")
    print("Classification Report:")
    print(report_str)
    print("Confusion Matrix:")
    print(cm)

    fold_results.append(
        {
            "accuracy": final_acc,
            "f1_score": final_f1,
            "auc": final_auc,
            "report": report_dict,
            "cm": cm
        }
    )

# --- Evaluation ---
# Summarize results across folds
accs = [r["accuracy"] for r in fold_results]
f1s = [r["f1_score"] for r in fold_results]
aucs = [r["auc"] for r in fold_results]

print("\n--- Cross-Validation Summary ---")
print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Mean F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# --- Saving Final Model ---
print("\n--- Training Final Model on Full Balanced Dataset ---")
final_model = get_model().to(device)
final_ds = FERDataset(df_balanced, transform=train_transform) # Use train transforms
final_loader = DataLoader(
    final_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
)
criterion = nn.CrossEntropyLoss()

# --- Phase 1: Train the Head on Full Data ---
print("\n--- Final Model - Phase 1: Training Head ---")
for param in final_model.features.parameters():
    param.requires_grad = False
optimizer_head_final = optim.AdamW(
    final_model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY
)
for epoch in range(EPOCHS_PHASE1):
    final_model.train()
    running_loss = 0.0
    start_time = time.time()
    for imgs, labels in final_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_head_final.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_head_final.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(final_loader.dataset)
    epoch_time = time.time() - start_time
    print(
        f"Final Phase 1 - Epoch {epoch+1}/{EPOCHS_PHASE1}, Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
    )


# --- Phase 2: Fine-tune the whole model on Full Data ---
print("\n--- Final Model - Phase 2: Fine-tuning Full Model ---")
for param in final_model.features.parameters():
    param.requires_grad = True
optimizer_full_final = optim.AdamW(
    [
        {
            "params": final_model.features.parameters(),
            "lr": LEARNING_RATE_BACKBONE,
        },
        {
            "params": final_model.classifier.parameters(),
            "lr": LEARNING_RATE_HEAD,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)
for epoch in range(EPOCHS_PHASE2):
    final_model.train()
    running_loss = 0.0
    start_time = time.time()
    for imgs, labels in final_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_full_final.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_full_final.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(final_loader.dataset)
    epoch_time = time.time() - start_time
    print(
        f"Final Phase 2 - Epoch {epoch+1}/{EPOCHS_PHASE2}, Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
    )

# Save model
os.makedirs("models", exist_ok=True)
save_path = "models/efficientnetb0_fer2013_happy_finetuned.pth"
torch.save(final_model.state_dict(), save_path)
print(f"\nFinal model saved to {save_path}")

# --- Optimization (Placeholder) ---
print("\n--- Optimization Notes ---")
print("The saved model is a standard PyTorch state dictionary.")
print("For near real-time performance as discussed in the methodology,")
print("further optimization steps like conversion to ONNX, TensorRT, or OpenVINO,")
print("and potentially quantization, would be necessary.")
print("These steps are typically performed after training.")

