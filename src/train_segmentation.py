# train_segmentation.py — Train Attention U-Net for stroke lesion segmentation
#
# Run from repo root:
#   python src/train_segmentation.py

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.attention_unet import AttentionUNet
from segmentation_dataset import StrokeSegmentationDataset, load_segmentation_ids
from losses import ComboLoss
from metrics import compute_all_metrics

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR    = "data/Segmentation_Data"   # must contain PNG/, MASKS/, labels.csv
MODELS_DIR  = "outputs/models"
BASE_CHANNELS = 16
BATCH_SIZE  = 8
EPOCHS      = 60
LR          = 1e-4
VAL_SPLIT   = 0.2
SEED        = 42

os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── DATA ──────────────────────────────────────────────────────────────────
train_ids, val_ids = load_segmentation_ids(DATA_DIR, val_split=VAL_SPLIT, seed=SEED)

train_ds = StrokeSegmentationDataset(DATA_DIR, train_ids, augment=True)
val_ds   = StrokeSegmentationDataset(DATA_DIR, val_ids, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ─── MODEL / LOSS / OPTIMIZER ──────────────────────────────────────────────
model = AttentionUNet(in_channels=1, base_channels=BASE_CHANNELS).to(device)
criterion = ComboLoss(bce_weight=0.3, pos_weight=10.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# NOTE on model selection: with ~88% of validation images having empty
# masks, a naive per-image Dice average is dominated by trivial "predict
# nothing" scores (empty vs empty = 1.0). That average looks good even when
# the model barely detects real lesions. Instead, we pool ALL validation
# pixels together (same method evaluate_segmentation.py uses) and select by
# balanced accuracy = (Sensitivity + Specificity) / 2. Raw Sensitivity alone
# would be gamed by a degenerate "predict everything as lesion" model
# (Sensitivity=1.0, Specificity=0.0) -- balanced accuracy only rewards a
# checkpoint that's good at BOTH catching real lesions and not over-flagging.
best_val_balanced_acc = 0.0
model_path = os.path.join(MODELS_DIR, "attention_unet.pth")

# ─── TRAIN LOOP ────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_ds)

    # ─── VALIDATION (pooled, not per-image averaged) ───────────────────
    model.eval()
    val_loss = 0.0
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item() * images.size(0)

            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    val_loss /= len(val_ds)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_masks, axis=0)
    metrics = compute_all_metrics(y_true, y_pred, threshold=0.5)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
          f"Dice: {metrics['Dice Score']:.4f} | IoU: {metrics['IoU']:.4f} | "
          f"Sens: {metrics['Sensitivity']:.4f} | Spec: {metrics['Specificity']:.4f}")

    balanced_acc = (metrics["Sensitivity"] + metrics["Specificity"]) / 2
    if balanced_acc > best_val_balanced_acc:
        best_val_balanced_acc = balanced_acc
        torch.save(model.state_dict(), model_path)
        print(f"  -> New best balanced accuracy ({balanced_acc:.4f}, "
              f"Sens={metrics['Sensitivity']:.4f} Spec={metrics['Specificity']:.4f}), saved to {model_path}")

print(f"\nTraining complete. Best val balanced accuracy: {best_val_balanced_acc:.4f}")
print(f"Model saved to: {model_path}")
