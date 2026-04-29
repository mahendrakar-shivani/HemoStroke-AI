# f1_score.py — F1 Score (Harmonic mean of Precision and Recall)

import numpy as np

def f1_score(y_true, y_pred, threshold=0.5):
    """
    F1 = 2 * Precision * Recall / (Precision + Recall)
    Balances false positives and false negatives.
    Range: 0 to 1 (higher = better overall performance)

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities or labels
        threshold: cutoff to convert probability to label
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
    fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    if (precision + recall) == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))