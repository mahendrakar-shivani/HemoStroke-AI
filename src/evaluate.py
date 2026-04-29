# evaluate.py — Evaluate any saved model with all 5 metrics

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CTScanDataset, load_dataset, get_transforms
from models  import MODEL_REGISTRY, get_model
from metrics import compute_all_metrics

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed/png"
MODELS_DIR  = "outputs/models"
PLOTS_DIR   = "outputs/plots"
BATCH_SIZE  = 32
SPLIT_RATIO = 0.8

os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD TEST DATA ──────────────────────────────────────────────────────────
image_paths, labels = load_dataset(DATA_DIR)
total      = len(image_paths)
train_size = int(SPLIT_RATIO * total)

np.random.seed(42)
indices   = list(range(total))
np.random.shuffle(indices)
test_idx  = indices[train_size:]

test_paths  = [image_paths[i] for i in test_idx]
test_labels = [labels[i]      for i in test_idx]

test_dataset = CTScanDataset(
    test_paths, test_labels,
    transform=get_transforms(augment=False)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_model(model_name):
    """Load and evaluate one model. Returns metrics dict."""

    model_file = os.path.join(
        MODELS_DIR,
        f"{model_name.replace(' ', '_').lower()}.pth"
    )

    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
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

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # All 5 metrics
    metrics = compute_all_metrics(y_true, y_pred)
    accuracy = 100 * sum(1 for p, l in zip(y_pred, y_true)
                         if (p >= 0.5) == bool(l)) / len(y_true)

    print(f"\n── {model_name} ──────────────────────────────────")
    print(f"  Accuracy    : {accuracy:.2f}%")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    # Confusion matrix
    cm   = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Stroke"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(
        PLOTS_DIR,
        f"{model_name.replace(' ', '_').lower()}_confusion.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {save_path}")

    return {"accuracy": accuracy, **metrics}


# ─── EVALUATE ALL AVAILABLE MODELS ───────────────────────────────────────────
all_results = {}

for model_name in MODEL_REGISTRY.keys():
    result = evaluate_model(model_name)
    if result:
        all_results[model_name] = result

# ─── FINAL COMPARISON TABLE ──────────────────────────────────────────────────
if all_results:
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON — ALL MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Acc':>7} {'Dice':>8} {'IoU':>8} "
          f"{'Sens':>8} {'Spec':>8} {'F1':>8}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<20} {r['accuracy']:>6.2f}% "
              f"{r['Dice Score']:>8.4f} "
              f"{r['IoU']:>8.4f} "
              f"{r['Sensitivity']:>8.4f} "
              f"{r['Specificity']:>8.4f} "
              f"{r['F1 Score']:>8.4f}")