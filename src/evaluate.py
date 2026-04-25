# evaluate.py — Confusion matrix + detailed metrics

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from torch.utils.data import DataLoader
import numpy as np

from dataset import CTScanDataset, load_dataset, get_transforms
from model   import StrokeCNN

DATA_DIR   = "data/processed/png"
MODEL_PATH = "outputs/stroke_cnn.pth"
BATCH_SIZE = 32
SPLIT_RATIO = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load same test split
image_paths, labels = load_dataset(DATA_DIR)
total      = len(image_paths)
train_size = int(SPLIT_RATIO * total)

np.random.seed(42)   # same seed = same split as training
indices    = list(range(total))
np.random.shuffle(indices)
test_idx   = indices[train_size:]

test_paths  = [image_paths[i] for i in test_idx]
test_labels = [labels[i]      for i in test_idx]

test_dataset = CTScanDataset(test_paths, test_labels,
                              transform=get_transforms(augment=False))
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = StrokeCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}\n")

# Run predictions
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images  = images.to(device)
        outputs = model(images)
        preds   = (outputs >= 0.5).float().cpu().squeeze()

        if preds.dim() == 0:
            preds = preds.unsqueeze(0)

        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())

# Report
print("── Classification Report ──────────────────────────────")
print(classification_report(all_labels, all_preds,
                             target_names=["Normal", "Stroke"]))

# Confusion Matrix
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Stroke"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("HemoStroke-AI — Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()
print("Saved → outputs/confusion_matrix.png")