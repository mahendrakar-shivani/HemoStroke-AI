# specificity.py — Specificity (True Negative Rate)

import numpy as np

def specificity(y_true, y_pred, threshold=0.5):
    """
    Specificity = TN / (TN + FP)
    = How many NORMAL scans did we correctly identify?
    High specificity = fewer false alarms.
    Range: 0 to 1 (higher = fewer false positives)

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities or labels
        threshold: cutoff to convert probability to label
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    tn = ((y_pred_bin == 0) & (y_true_bin == 0)).sum()
    fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()

    if (tn + fp) == 0:
        return 0.0

    return float(tn / (tn + fp))