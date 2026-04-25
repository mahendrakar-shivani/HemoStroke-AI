# train.py — Full training pipeline with class imbalance handling

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np

from dataset import CTScanDataset, load_dataset, get_transforms
from model   import StrokeCNN, count_parameters

# Fix random seed so results are reproducible
np.random.seed(42)
torch.manual_seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed/png"
SAVE_PATH   = "outputs/stroke_cnn.pth"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.0001
SPLIT_RATIO = 0.8

# ─── DEVICE ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("TIP: Training on CPU will be slow. Consider Google Colab for GPU.")

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
image_paths, labels = load_dataset(DATA_DIR)

# Split indices: 80% train, 20% test
total       = len(image_paths)
train_size  = int(SPLIT_RATIO * total)
test_size   = total - train_size

indices     = list(range(total))
np.random.shuffle(indices)
train_idx   = indices[:train_size]
test_idx    = indices[train_size:]

train_paths  = [image_paths[i] for i in train_idx]
train_labels = [labels[i]      for i in train_idx]
test_paths   = [image_paths[i] for i in test_idx]
test_labels  = [labels[i]      for i in test_idx]

# ─── HANDLE CLASS IMBALANCE ──────────────────────────────────────────────────
# Give Stroke images higher chance of being picked each batch
# so model sees equal Normal and Stroke during training
normal_count = train_labels.count(0)
stroke_count = train_labels.count(1)
total_train  = len(train_labels)

# Weight: rarer class gets higher weight
weight_for_normal = total_train / (2 * normal_count)
weight_for_stroke = total_train / (2 * stroke_count)

sample_weights = [
    weight_for_normal if lbl == 0 else weight_for_stroke
    for lbl in train_labels
]

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

print(f"\nClass weights:")
print(f"  Normal weight : {weight_for_normal:.4f}")
print(f"  Stroke weight : {weight_for_stroke:.4f}")

# ─── DATASETS & LOADERS ──────────────────────────────────────────────────────
train_transform = get_transforms(augment=True)   # augmented for training
test_transform  = get_transforms(augment=False)  # clean for testing

train_dataset = CTScanDataset(train_paths, train_labels, transform=train_transform)
test_dataset  = CTScanDataset(test_paths,  test_labels,  transform=test_transform)

# Use sampler instead of shuffle=True for balanced batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataset split:")
print(f"  Train : {train_size} images")
print(f"  Test  : {test_size}  images")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches per epoch: {len(train_loader)}")

# ─── MODEL ───────────────────────────────────────────────────────────────────
model = StrokeCNN().to(device)
count_parameters(model)

# pos_weight handles imbalance in loss function too (extra safety)
pos_weight = torch.tensor([normal_count / stroke_count]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# NOTE: Since we use BCEWithLogitsLoss, remove Sigmoid from model output
# OR keep BCELoss + Sigmoid — we'll keep Sigmoid + BCELoss for simplicity:
criterion  = nn.BCELoss()

optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
train_losses     = []
test_accuracies  = []
best_accuracy    = 0.0

print("\n" + "="*60)
print("  Starting Training — HemoStroke-AI CNN")
print("="*60)

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN PHASE ──────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for batch_idx, (images, lbls) in enumerate(train_loader):
        images = images.to(device)
        lbls   = lbls.to(device).unsqueeze(1)  # shape: (batch, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── EVAL PHASE ───────────────────────────────────────────────
    model.eval()
    correct = total_correct = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device)
            lbls   = lbls.to(device).unsqueeze(1)

            outputs   = model(images)
            predicted = (outputs >= 0.5).float()

            total_correct += lbls.size(0)
            correct       += (predicted == lbls).sum().item()

            # For precision/recall tracking
            tp += ((predicted == 1) & (lbls == 1)).sum().item()
            fp += ((predicted == 1) & (lbls == 0)).sum().item()
            tn += ((predicted == 0) & (lbls == 0)).sum().item()
            fn += ((predicted == 0) & (lbls == 1)).sum().item()

    accuracy  = 100 * correct / total_correct
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)   # = sensitivity
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    test_accuracies.append(accuracy)

    print(f"\nEpoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} | "
          f"Acc: {accuracy:.2f}% | "
          f"Precision: {precision:.3f} | "
          f"Recall: {recall:.3f} | "
          f"F1: {f1:.3f}")
    print("-"*60)

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved (accuracy: {accuracy:.2f}%)")

    scheduler.step()  # reduce LR every 3 epochs

print("="*60)
print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
print(f"Model saved → {SAVE_PATH}")

# ─── PLOT CURVES ─────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, marker='o', color='tomato', linewidth=2)
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS+1), test_accuracies, marker='o', color='steelblue', linewidth=2)
ax2.axhline(y=best_accuracy, color='green', linestyle='--', label=f'Best: {best_accuracy:.1f}%')
ax2.set_title("Test Accuracy per Epoch")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("HemoStroke-AI — CNN Training Results", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150)
print("Training curves saved → outputs/training_curves.png")