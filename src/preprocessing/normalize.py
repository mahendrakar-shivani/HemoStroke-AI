# normalize.py — Normalize CT scan pixel values

import numpy as np

def normalize_minmax(image):
    """
    Min-Max normalization → scales to [0, 1].
    Standard for neural network input.
    """
    img_min = image.min()
    img_max = image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - img_min) / (img_max - img_min)).astype(np.float32)

def normalize_zscore(image):
    """
    Z-score normalization → mean=0, std=1.
    Better for DICOM Hounsfield Units.
    """
    mean = image.mean()
    std  = image.std()
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mean) / std).astype(np.float32)

def apply_brain_window(image, window_level=40, window_width=80):
    """
    Apply brain window for CT scans.
    WL=40, WW=80 is standard for ischemic stroke visibility.
    Clips Hounsfield Units to brain tissue range.
    """
    lower = window_level - window_width // 2
    upper = window_level + window_width // 2
    windowed = np.clip(image, lower, upper)
    return normalize_minmax(windowed)

def normalize_image(image, method="minmax"):
    """
    Master normalize function.
    method: 'minmax', 'zscore', or 'brain_window'
    """
    if method == "minmax":
        return normalize_minmax(image)
    elif method == "zscore":
        return normalize_zscore(image)
    elif method == "brain_window":
        return apply_brain_window(image)
    else:
        raise ValueError(f"Unknown method: {method}")