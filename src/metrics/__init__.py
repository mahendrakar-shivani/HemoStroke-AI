# metrics/__init__.py — Import all metrics in one line

from metrics.dice_score  import dice_score
from metrics.iou_score   import iou_score
from metrics.sensitivity import sensitivity
from metrics.specificity import specificity
from metrics.f1_score    import f1_score

def compute_all_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute all 5 metrics at once.
    Returns a dictionary with all results.
    
    Usage:
        results = compute_all_metrics(y_true, y_pred)
        print(results)
    """
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "Dice Score"  : round(dice_score(y_true, y_pred, threshold), 4),
        "IoU"         : round(iou_score(y_true, y_pred, threshold), 4),
        "Sensitivity" : round(sensitivity(y_true, y_pred, threshold), 4),
        "Specificity" : round(specificity(y_true, y_pred, threshold), 4),
        "F1 Score"    : round(f1_score(y_true, y_pred, threshold), 4),
    }