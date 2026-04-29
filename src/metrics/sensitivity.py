# sensitivity.py — Sensitivity (True Positive Rate / Recall)

import numpy as np

def sensitivity(y_true, y_pred, threshold=0.5):
    """
    Sensitivity = TP / (TP + FN)
    = How many REAL strokes did we correctly detect?
    Most important metric in medical AI.
    Range: 0 to 1 (higher = fewer missed strokes)

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities or labels
        threshold: cutoff to convert probability to label
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
    fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()

    if (tp + fn) == 0:
        return 0.0

    return float(tp / (tp + fn))