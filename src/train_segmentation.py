# train_segmentation.py — Train Attention U-Net for stroke lesion segmentation
#
# Run from repo root:
#   python src/train_segmentation.py

import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.attention_unet import AttentionUNet
from segmentation_dataset import StrokeSegmentationDataset, load_segmentation_ids
from losses import ComboLoss
from metrics import dice_score, iou_score

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR    = "data/Segmentation_Data"   # must contain PNG/, MASKS/, labels.csv
MODELS_DIR  = "outputs/models"
BASE_CHANNELS = 16     # small model — dataset is only ~200 images
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─── MODEL / LOSS / OPTIMIZER ──────────────────────────────────────────────
model = AttentionUNet(in_channels=1, base_channels=BASE_CHANNELS).to(device)
criterion = ComboLoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

best_val_dice = 0.0
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

    # ─── VALIDATION ─────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item() * images.size(0)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for p, m in zip(preds_np, masks_np):
                dice_scores.append(dice_score(m, p))
                iou_scores.append(iou_score(m, p))

    val_loss /= len(val_ds)
    val_dice = sum(dice_scores) / len(dice_scores)
    val_iou = sum(iou_scores) / len(iou_scores)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
          f"val_dice: {val_dice:.4f} | val_iou: {val_iou:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), model_path)
        print(f"  -> New best val_dice ({val_dice:.4f}), saved to {model_path}")

print(f"\nTraining complete. Best val Dice: {best_val_dice:.4f}")
print(f"Model saved to: {model_path}")
