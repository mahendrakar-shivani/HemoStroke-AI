# dice_score.py — Dice Similarity Coefficient

import numpy as np

def dice_score(y_true, y_pred, threshold=0.5):
    """
    Dice Score = 2 * |A ∩ B| / (|A| + |B|)
    Measures overlap between predicted and true mask.
    Range: 0 (no overlap) to 1 (perfect overlap)

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities or labels
        threshold: cutoff to convert probability to label
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    intersection = (y_true_bin * y_pred_bin).sum()
    union        = y_true_bin.sum() + y_pred_bin.sum()

    if union == 0:
        return 1.0  # both empty = perfect match

    return float(2.0 * intersection / union)