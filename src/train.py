# train.py — Train all 3 CNN models and save each one

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np

# Add src/ to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset         import CTScanDataset, load_dataset, get_transforms
from models          import MODEL_REGISTRY
from metrics         import compute_all_metrics

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed/png"
SAVE_DIR    = "outputs/models"
PLOTS_DIR   = "outputs/plots"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.0001
SPLIT_RATIO = 0.8

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
image_paths, labels = load_dataset(DATA_DIR)
total      = len(image_paths)
train_size = int(SPLIT_RATIO * total)
test_size  = total - train_size

np.random.seed(42)
torch.manual_seed(42)
indices    = list(range(total))
np.random.shuffle(indices)
train_idx  = indices[:train_size]
test_idx   = indices[train_size:]

train_paths  = [image_paths[i] for i in train_idx]
train_labels = [labels[i]      for i in train_idx]
test_paths   = [image_paths[i] for i in test_idx]
test_labels  = [labels[i]      for i in test_idx]

# Handle class imbalance
normal_count = train_labels.count(0)
stroke_count = train_labels.count(1)
total_train  = len(train_labels)
w_normal     = total_train / (2 * normal_count)
w_stroke     = total_train / (2 * stroke_count)
sample_weights = [w_normal if l == 0 else w_stroke for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_dataset = CTScanDataset(train_paths, train_labels, transform=get_transforms(augment=True))
test_dataset  = CTScanDataset(test_paths,  test_labels,  transform=get_transforms(augment=False))
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size} | Test: {test_size}")


# ─── TRAIN ONE MODEL ─────────────────────────────────────────────────────────
def train_model(model_name, model):
    """Train a single model and save it."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model      = model.to(device)
    criterion  = nn.BCELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_losses    = []
    test_accuracies = []
    best_accuracy   = 0.0
    best_preds      = []
    best_labels     = []

    for epoch in range(1, EPOCHS + 1):

        # ── Train ────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for images, lbls in train_loader:
            images = images.to(device)
            lbls   = lbls.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── Evaluate ─────────────────────────────────────────────
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for images, lbls in test_loader:
                images  = images.to(device)
                outputs = model(images)
                preds   = outputs.cpu().squeeze().tolist()
                labs    = lbls.tolist()
                if isinstance(preds, float): preds = [preds]
                if isinstance(labs,  float): labs  = [labs]
                all_preds.extend(preds)
                all_labels.extend(labs)

        # Compute accuracy
        correct  = sum(1 for p, l in zip(all_preds, all_labels)
                       if (p >= 0.5) == bool(l))
        accuracy = 100 * correct / len(all_labels)
        test_accuracies.append(accuracy)

        # Compute all 5 metrics
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.2f}% | "
              f"Dice: {metrics['Dice Score']:.3f} | "
              f"IoU: {metrics['IoU']:.3f} | "
              f"Sens: {metrics['Sensitivity']:.3f} | "
              f"Spec: {metrics['Specificity']:.3f} | "
              f"F1: {metrics['F1 Score']:.3f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_preds    = all_preds.copy()
            best_labels   = all_labels.copy()
            save_path = os.path.join(
                SAVE_DIR,
                f"{model_name.replace(' ', '_').lower()}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model ({accuracy:.2f}%) → {save_path}")

        scheduler.step()

    # ── Plot training curves ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, EPOCHS+1), train_losses,
             marker='o', color='tomato', linewidth=2)
    ax1.set_title(f"{model_name} — Training Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, EPOCHS+1), test_accuracies,
             marker='o', color='steelblue', linewidth=2)
    ax2.axhline(y=best_accuracy, color='green',
                linestyle='--', label=f'Best: {best_accuracy:.1f}%')
    ax2.set_title(f"{model_name} — Test Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(
        PLOTS_DIR,
        f"{model_name.replace(' ', '_').lower()}_curves.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {plot_path}")

    # Final metrics
    final_metrics = compute_all_metrics(
        np.array(best_labels),
        np.array(best_preds)
    )
    print(f"\n  Final Metrics for {model_name}:")
    for k, v in final_metrics.items():
        print(f"    {k}: {v}")

    return best_accuracy, final_metrics


# ─── TRAIN ALL MODELS ────────────────────────────────────────────────────────
results = {}

for model_name, ModelClass in MODEL_REGISTRY.items():
    model = ModelClass()
    best_acc, metrics = train_model(model_name, model)
    results[model_name] = {"accuracy": best_acc, **metrics}

# ─── COMPARISON SUMMARY ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<20} {'Accuracy':>10} {'Dice':>8} {'IoU':>8} "
      f"{'Sens':>8} {'Spec':>8} {'F1':>8}")
print("-" * 72)

for name, r in results.items():
    print(f"{name:<20} {r['accuracy']:>9.2f}% "
          f"{r['Dice Score']:>8.4f} "
          f"{r['IoU']:>8.4f} "
          f"{r['Sensitivity']:>8.4f} "
          f"{r['Specificity']:>8.4f} "
          f"{r['F1 Score']:>8.4f}")

# ─── COMPARISON PLOT ─────────────────────────────────────────────────────────
model_names  = list(results.keys())
metric_names = ["Dice Score", "IoU", "Sensitivity", "Specificity", "F1 Score"]
x = np.arange(len(metric_names))
width = 0.25
colors = ["steelblue", "tomato", "seagreen"]

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metric_names]
    ax.bar(x + i * width, vals, width, label=name,
           color=color, alpha=0.85)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("HemoStroke-AI — Model Comparison (All 5 Metrics)")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
plt.close()
print(f"\nComparison chart saved → {PLOTS_DIR}/model_comparison.png")
print("All models trained successfully!")