   # imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Use updated torch.amp components
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from PIL import Image
import os
import random
import copy
import time
import logging # Logging
from tqdm import tqdm # Use standard tqdm for console

# --- Configuration ---
# Path to your FER2013 CSV file
CSV_PATH = "fer2013.csv"
# Model/Checkpoint saving directory
MODEL_DIR = "models_checkpointed"
# Log file path
LOG_FILE = "training_log.log"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE), # Output to file
        logging.StreamHandler()        # Output to console
    ]
)
logging.info(f"Logging to console and {LOG_FILE}")

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
        torch.backends.cudnn.benchmark = False # Can impact performance, but needed for full determinism
    logging.info(f"Random seed set to {seed}")

SEED = 42
set_seed(SEED)

# --- Training Hyperparameters ---
BATCH_SIZE = 16
EPOCHS_PHASE1 = 3 # Epochs for training the head only
EPOCHS_PHASE2 = 7 # Epochs for fine-tuning the whole model
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2
K_FOLDS = 5
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-2
SCHEDULER_PATIENCE = 3 # LR scheduler patience
SCHEDULER_FACTOR = 0.1 # LR reduction factor
EARLY_STOPPING_PATIENCE = 5 # Early stopping patience based on F1
DROPOUT_RATE = 0.5 # As specified in methodology

# --- Load Data ---
logging.info(f"Loading data from {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    logging.error(f"Error: CSV file not found at {CSV_PATH}")
    exit() # Exit if data file is missing
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    exit()

# Separate PrivateTest set
df_test = df[df["Usage"] == "PrivateTest"].copy()
df_train_val = df[df["Usage"].isin(["Training", "PublicTest"])].copy()
logging.info(f"Loaded {len(df_train_val)} samples for Training/Validation")
logging.info(f"Loaded {len(df_test)} samples for Final Testing")

# Map emotion: 3 (happy) -> 1, others -> 0
df_train_val["label"] = (df_train_val["emotion"] == 3).astype(int)
df_test["label"] = (df_test["emotion"] == 3).astype(int)

# Calculate Class Weights for the training/validation data
class_counts = df_train_val["label"].value_counts().sort_index()
if len(class_counts) < 2:
     logging.warning("Only one class found in training/validation data. Class weighting might not work as expected.")
     # Handle case with only one class if necessary, e.g., set weights to 1
     class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32) # Default or adjust
elif 0 in class_counts and class_counts[0] == 0 or 1 in class_counts and class_counts[1] == 0:
    logging.warning("One class has zero samples in training/validation data. Adjusting weights.")
    # Adjust weights or handle appropriately, e.g., equal weights
    class_weights_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32)
else:
    total_samples = len(df_train_val)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(
        class_weights.values, dtype=torch.float32
    )
logging.info(f"Class counts (Train/Val): {class_counts.to_dict()}")
logging.info(f"Calculated class weights: {class_weights_tensor.numpy()}")

# --- Dataset Class ---
class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Use np.fromiter which is generally safer/faster for text
            pixels = np.fromiter(
                row["pixels"].split(" "), dtype=np.uint8
            ).reshape(48, 48)
        except Exception as e:
            logging.error(f"Error processing row {idx} with label {row.get('label', 'N/A')}: {row['pixels'][:50]}... Error: {e}")
            # Option 1: Re-raise the exception to stop execution
            raise e
            # Option 2: Return None or a placeholder to skip the item (requires handling in DataLoader collation)
            # return None, None
            # Option 3: Return a dummy item (might skew results if not handled)
            # dummy_img = torch.zeros((3, 224, 224)) # Match transform output shape
            # dummy_label = -1 # Or some indicator label
            # return dummy_img, dummy_label

        img = Image.fromarray(pixels).convert("RGB")
        label = row["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Transforms ---
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- Model Definition ---
def get_model(dropout_rate=DROPOUT_RATE):
    # Use weights= argument for modern torchvision
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(num_ftrs, 2), # Binary classification
    )
    return model

# --- Training & Evaluation Functions ---
def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, phase="Head"
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Training Epoch ({phase})", leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        # Use updated torch.amp.autocast syntax
        with autocast(device_type=device.type, enabled=(scaler is not None)):
            try:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            except Exception as e:
                logging.error(f"Error during forward/loss calculation in training batch {i}: {e}")
                # Optionally skip batch or raise error
                continue # Skip this batch

        if scaler: # If using mixed precision
            try:
                scaler.scale(loss).backward()
                # Unscales gradients and calls optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
            except Exception as e:
                 logging.error(f"Error during backward/step with scaler in training batch {i}: {e}")
                 # Consider skipping optimizer step or handling differently
                 optimizer.zero_grad(set_to_none=True) # Ensure grads are cleared even if step fails
                 continue
        else: # Standard precision
            try:
                loss.backward()
                optimizer.step()
            except Exception as e:
                 logging.error(f"Error during backward/step without scaler in training batch {i}: {e}")
                 optimizer.zero_grad(set_to_none=True)
                 continue

        # Ensure loss is a valid number before accumulating
        if not torch.isnan(loss) and not torch.isinf(loss):
            running_loss += loss.item() * imgs.size(0)
        else:
            logging.warning(f"NaN or Inf loss detected in training batch {i}. Skipping accumulation.")

        pbar.set_postfix(loss=loss.item() if not torch.isnan(loss) else float('nan'))

    # Avoid division by zero if loader is empty
    if not loader.dataset:
        return 0.0
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    all_probs = []
    running_loss = 0.0
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            try:
                # Autocast can optionally be used here too for consistency/speed
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    running_loss += loss.item() * imgs.size(0)
                else:
                    logging.warning(f"NaN or Inf loss detected in evaluation batch {i}. Skipping accumulation.")

                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
            except Exception as e:
                logging.error(f"Error during evaluation batch {i}: {e}")
                continue # Skip batch on error

    # Avoid division by zero if loader is empty or all batches failed
    if not loader.dataset or not all_targets:
        logging.warning("Evaluation dataset empty or all batches failed.")
        # Return default metrics or handle as appropriate
        return {
            "loss": float('inf'), "accuracy": 0.0, "f1": 0.0, "precision": 0.0,
            "recall": 0.0, "auc": 0.0, "cm": np.zeros((2, 2)), "report": {},
            "preds": np.array([]), "targets": np.array([])
        }

    val_loss = running_loss / len(all_targets) # Use length of collected targets

    # Ensure targets and preds are numpy arrays for sklearn metrics
    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    val_acc = accuracy_score(all_targets_np, all_preds_np)
    val_f1 = f1_score(all_targets_np, all_preds_np, average="binary", zero_division=0)
    val_prec = precision_score(all_targets_np, all_preds_np, average="binary", zero_division=0)
    val_rec = recall_score(all_targets_np, all_preds_np, average="binary", zero_division=0)

    val_auc = 0.0 # Default AUC
    try:
        # Ensure there are samples for both classes for AUC calculation
        if len(np.unique(all_targets_np)) > 1:
             val_auc = roc_auc_score(all_targets_np, all_probs_np)
        else:
             logging.warning("AUC calculation skipped: only one class present in evaluation targets.")
    except ValueError as e:
        logging.warning(f"AUC calculation failed: {e}")

    cm = confusion_matrix(all_targets_np, all_preds_np)
    try:
        report_dict = classification_report(
                all_targets_np, all_preds_np, target_names=["Not Happy", "Happy"], output_dict=True, zero_division=0
            )
    except Exception as e:
        logging.error(f"Error generating classification report: {e}")
        report_dict = {} # Provide an empty dict if report fails

    metrics = {
        "loss": val_loss,
        "accuracy": val_acc,
        "f1": val_f1,
        "precision": val_prec,
        "recall": val_rec,
        "auc": val_auc,
        "cm": cm,
        "report": report_dict,
        "preds": all_preds_np,     # Return predictions
        "targets": all_targets_np  # Return targets
    }
    return metrics


# --- Main K-Fold Cross-Validation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
# Set num_workers. Set to 0 for debugging dataloader/multiprocessing issues.
num_workers = 4 if os.name != "nt" and torch.cuda.is_available() else 0
logging.info(f"Using {num_workers} workers for DataLoaders")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
X = df_train_val.index.values # Use index for splitting
y = df_train_val["label"].values

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    logging.info(f"--- Starting Fold {fold+1}/{K_FOLDS} ---")
    fold_start_time = time.time()

    train_df = df_train_val.iloc[train_idx]
    val_df = df_train_val.iloc[val_idx]

    train_ds = FERDataset(train_df, transform=train_transform)
    val_ds = FERDataset(val_df, transform=val_test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True # drop_last can help with batchnorm issues on small final batches
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = get_model().to(device)
    # Use class weights in criterion
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    # Use updated torch.amp.GradScaler syntax
    scaler = GradScaler('cuda') if device.type == "cuda" else None
    if scaler:
        logging.info("Using Mixed Precision (AMP) with torch.amp.GradScaler")

    # --- Phase 1: Train the Head ---
    logging.info("--- Phase 1: Training Head ---")
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    # Ensure classifier parameters require grad (redundant if model defined correctly, but safe)
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Optimizer for head only - ensure it only gets parameters requiring grad
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_HEAD,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS_PHASE1):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, phase="Head"
        )
        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"Fold {fold+1} Phase 1 - Epoch {epoch+1}/{EPOCHS_PHASE1}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s"
        )

    # --- Phase 2: Fine-tune the whole model ---
    logging.info("--- Phase 2: Fine-tuning Full Model ---")
    # Unfreeze backbone
    for param in model.features.parameters():
        param.requires_grad = True

    # Optimizer with differential learning rates
    optimizer = optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": LEARNING_RATE_BACKBONE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": LEARNING_RATE_HEAD, # Keep head LR higher initially
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=False # Quieter scheduler
    )

    # Early Stopping variables
    best_val_f1 = -1.0 # Initialize to handle F1=0 case
    patience_counter = 0
    best_epoch = -1
    checkpoint_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold+1}.pth")

    for epoch in range(EPOCHS_PHASE2):
        epoch_start_time = time.time()
        current_epoch_total = EPOCHS_PHASE1 + epoch + 1

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, phase="Full"
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time
        # Log current learning rate(s)
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        logging.info(
            f"Fold {fold+1} Phase 2 - Epoch {epoch+1}/{EPOCHS_PHASE2} (Total: {current_epoch_total}), "
            f"LR: {current_lrs}, Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
            # f"Val Prec: {val_metrics['precision']:.4f}, Val Rec: {val_metrics['recall']:.4f}, " # Optional: reduce log verbosity
            f"Val AUC: {val_metrics['auc']:.4f}, Time: {epoch_time:.2f}s"
        )

        # Step the scheduler based on validation loss
        scheduler.step(val_metrics['loss'])

        # Early Stopping Check based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_epoch = current_epoch_total
            # Save checkpoint (model, optimizer, epoch, scheduler)
            try:
                torch.save({
                    'epoch': current_epoch_total,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state too
                    'loss': val_metrics['loss'],
                    'f1': best_val_f1,
                }, checkpoint_path)
                logging.info(
                    f"  -> New best F1: {best_val_f1:.4f} at epoch {best_epoch}. Checkpoint saved to {checkpoint_path}"
                )
            except Exception as e:
                 logging.error(f"Error saving checkpoint: {e}")
        else:
            patience_counter += 1
            logging.info(f"  -> F1 did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logging.info(f"  -> Early stopping triggered at epoch {current_epoch_total}.")
            break

    # Load best model state for final evaluation for this fold
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading best model from {checkpoint_path} (Epoch {best_epoch}, F1: {best_val_f1:.4f})")
        try:
            # Load checkpoint onto the correct device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Optionally load optimizer and scheduler if needed for resuming, but not for final eval
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Using last model state.")
            # Fallback to using the model state from the last epoch if loading fails
    else:
        logging.warning("No best model checkpoint found for this fold. Using last model state.")

    # Final Validation for the fold using the best loaded model
    logging.info(f"--- Evaluating Best Model for Fold {fold+1} ---")
    final_fold_metrics = evaluate(model, val_loader, criterion, device)

    logging.info(f"Fold {fold+1} Final Validation Results (Best Model):")
    logging.info(f"  Accuracy:  {final_fold_metrics['accuracy']:.4f}")
    logging.info(f"  F1 Score:  {final_fold_metrics['f1']:.4f}")
    logging.info(f"  Precision: {final_fold_metrics['precision']:.4f}")
    logging.info(f"  Recall:    {final_fold_metrics['recall']:.4f}")
    logging.info(f"  AUC:       {final_fold_metrics['auc']:.4f}")
    logging.info(f"  Loss:      {final_fold_metrics['loss']:.4f}")
    logging.info("  Confusion Matrix:")
    logging.info(f"\n{final_fold_metrics['cm']}")

    fold_results.append(final_fold_metrics)
    fold_time = time.time() - fold_start_time
    logging.info(f"--- Fold {fold+1} completed in {fold_time:.2f}s ---")


# --- Cross-Validation Summary ---
logging.info("--- Cross-Validation Summary ---")
if fold_results: # Check if any folds completed
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]

    logging.info(f"Mean Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logging.info(f"Mean F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logging.info(f"Mean Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    logging.info(f"Mean Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    logging.info(f"Mean AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
else:
    logging.warning("No fold results to summarize.")

# --- Training Final Model on Full Train/Val Data ---
logging.info("--- Training Final Model on Full Train/Val Dataset ---")
final_model = get_model().to(device)
final_ds = FERDataset(df_train_val, transform=train_transform) # Use train transforms
final_loader = DataLoader(
    final_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
)
criterion_final = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
scaler_final = GradScaler('cuda') if device.type == "cuda" else None

# --- Final Model - Phase 1: Train the Head ---
logging.info("--- Final Model - Phase 1: Training Head ---")
for param in final_model.features.parameters():
    param.requires_grad = False
for param in final_model.classifier.parameters():
        param.requires_grad = True
optimizer_head_final = optim.AdamW(
    filter(lambda p: p.requires_grad, final_model.parameters()),
    lr=LEARNING_RATE_HEAD,
    weight_decay=WEIGHT_DECAY
)
for epoch in range(EPOCHS_PHASE1):
    epoch_start_time = time.time()
    train_loss = train_one_epoch(
        final_model, final_loader, criterion_final, optimizer_head_final, device, scaler_final, phase="Head"
    )
    epoch_time = time.time() - epoch_start_time
    logging.info(
        f"Final Phase 1 - Epoch {epoch+1}/{EPOCHS_PHASE1}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s"
    )

# --- Final Model - Phase 2: Fine-tune the whole model ---
logging.info("--- Final Model - Phase 2: Fine-tuning Full Model ---")
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
# No scheduler or early stopping needed for final training run
for epoch in range(EPOCHS_PHASE2):
    epoch_start_time = time.time()
    train_loss = train_one_epoch(
        final_model, final_loader, criterion_final, optimizer_full_final, device, scaler_final, phase="Full"
    )
    epoch_time = time.time() - epoch_start_time
    logging.info(
        f"Final Phase 2 - Epoch {epoch+1}/{EPOCHS_PHASE2}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s"
    )

# Save final model (only state_dict needed for inference)
final_model_save_path = os.path.join(MODEL_DIR, "efficientnetb0_fer2013_happy_final.pth")
try:
    torch.save(final_model.state_dict(), final_model_save_path)
    logging.info(f"Final model state_dict saved to {final_model_save_path}")
except Exception as e:
    logging.error(f"Error saving final model: {e}")


# --- Final Evaluation on Test Set ---
logging.info("--- Evaluating Final Model on PrivateTest Set ---")
if len(df_test) > 0:
    test_ds = FERDataset(df_test, transform=val_test_transform)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Load the final trained model
    try:
        # Ensure the model is created before loading state_dict
        eval_model = get_model().to(device)
        eval_model.load_state_dict(torch.load(final_model_save_path, map_location=device))
        logging.info(f"Successfully loaded final model from {final_model_save_path}")

        # Evaluate on the test set using the loaded model
        test_metrics = evaluate(eval_model, test_loader, criterion_final, device) # Use same criterion setup

        logging.info("Final Test Set Results:")
        logging.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logging.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
        logging.info(f"  Precision: {test_metrics['precision']:.4f}")
        logging.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logging.info(f"  AUC:       {test_metrics['auc']:.4f}")
        logging.info(f"  Loss:      {test_metrics['loss']:.4f}")
        logging.info("  Confusion Matrix:")
        logging.info(f"\n{test_metrics['cm']}")
        logging.info("  Classification Report:")
        # Use the returned targets/preds from evaluate for the report
        if test_metrics['targets'].size > 0: # Check if evaluation produced results
             logging.info(f"\n{classification_report(test_metrics['targets'], test_metrics['preds'], target_names=['Not Happy', 'Happy'], zero_division=0)}")
        else:
             logging.warning("No predictions generated during final test evaluation.")

    except FileNotFoundError:
        logging.error(f"Final model file not found at {final_model_save_path}. Cannot evaluate on test set.")
    except Exception as e:
        logging.error(f"Error during final evaluation on test set: {e}", exc_info=True) # Log traceback

else:
    logging.warning("PrivateTest set is empty or could not be loaded. Skipping final evaluation.")


# --- Optimization Notes ---
logging.info("--- Optimization Notes ---")
logging.info("The saved final model is a standard PyTorch state dictionary.")
logging.info("For near real-time performance as discussed in the methodology,")
logging.info("further optimization steps like conversion to ONNX, TensorRT, or OpenVINO,")
logging.info("and potentially quantization, would be necessary after training.")

logging.info("--- Script Finished ---")