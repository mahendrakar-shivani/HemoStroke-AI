# evaluate_segmentation.py — Evaluate the trained Attention-UNet on the
# held-out validation split, using the same 5 metrics as the classifiers.
#
# Run from repo root:
#   python src/evaluate_segmentation.py

import os
import sys
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.attention_unet import AttentionUNet
from segmentation_dataset import StrokeSegmentationDataset, load_segmentation_ids
from metrics import compute_all_metrics

DATA_DIR    = "data/Segmentation_Data"
MODEL_PATH  = "outputs/models/attention_unet.pth"
METRICS_DIR = "outputs/metrics"
BASE_CHANNELS = 16
VAL_SPLIT   = 0.2
SEED        = 42

os.makedirs(METRICS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_ids = load_segmentation_ids(DATA_DIR, val_split=VAL_SPLIT, seed=SEED)
val_ds = StrokeSegmentationDataset(DATA_DIR, val_ids, augment=False)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

model = AttentionUNet(in_channels=1, base_channels=BASE_CHANNELS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds, all_masks = [], []
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        preds = model(images).cpu().numpy()
        all_preds.append(preds)
        all_masks.append(masks.numpy())

import numpy as np
y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_masks, axis=0)

results = compute_all_metrics(y_true, y_pred, threshold=0.5)
print("\nAttention U-Net — validation set (pixel-level):")
for k, v in results.items():
    print(f"  {k:<12}: {v:.4f}")

metrics_path = os.path.join(METRICS_DIR, "segmentation_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({"Attention U-Net": results}, f, indent=2)
print(f"\nSaved metrics -> {metrics_path}")
