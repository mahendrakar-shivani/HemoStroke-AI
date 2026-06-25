# skull_strip.py — Remove skull from CT scan, keep only brain

import cv2
import numpy as np

def threshold_skull_strip(image):
    """
    Simple skull stripping using thresholding + morphology.
    Fast, works well for standard CT scans.
    """
    img_uint8 = (image * 255).astype(np.uint8)

    # Threshold to separate brain from background
    _, binary = cv2.threshold(img_uint8, 10, 255, cv2.THRESH_BINARY)

    # Morphological close → fills small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Morphological open → removes small noise blobs
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask
    result = image * (mask / 255.0)
    return result.astype(np.float32)

def contour_skull_strip(image):
    """
    Skull stripping using intensity-based skull removal.
    Uses connected components (not filled contours) to avoid
    incorrectly re-including the skull-shaped gap as solid.
    """
    img_uint8 = (image * 255).astype(np.uint8)

    _, brain_only = cv2.threshold(img_uint8, 200, 255, cv2.THRESH_TOZERO_INV)
    _, binary = cv2.threshold(brain_only, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels = cv2.connectedComponents(cleaned)
    if num_labels <= 1:
        return image

    sizes = [np.sum(labels == i) for i in range(1, num_labels)]
    largest_label = 1 + int(np.argmax(sizes))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    result = image * (mask / 255.0)
    return result.astype(np.float32)

def skull_strip(image, method="contour"):
    """
    Master skull stripping function.
    method: 'threshold' or 'contour'
    """
    if method == "threshold":
        return threshold_skull_strip(image)
    elif method == "contour":
        return contour_skull_strip(image)
    else:
        raise ValueError(f"Unknown method: {method}")