# iou_score.py — Intersection over Union

import numpy as np

def iou_score(y_true, y_pred, threshold=0.5):
    """
    IoU = |A ∩ B| / |A ∪ B|
    Measures segmentation boundary accuracy.
    Range: 0 (no overlap) to 1 (perfect overlap)

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities or labels
        threshold: cutoff to convert probability to label
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    intersection = (y_true_bin * y_pred_bin).sum()
    union        = y_true_bin.sum() + y_pred_bin.sum() - intersection

    if union == 0:
        return 1.0

    return float(intersection / union)