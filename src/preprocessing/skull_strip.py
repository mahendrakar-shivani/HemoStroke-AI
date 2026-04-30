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
    Improved contour-based skull stripping.
    Uses erosion to better separate skull from brain tissue.
    """
    img_uint8 = (image * 255).astype(np.uint8)

    # Higher threshold to exclude bright skull bone
    _, binary = cv2.threshold(img_uint8, 20, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    # Keep largest contour
    largest = max(contours, key=cv2.contourArea)
    mask    = np.zeros_like(img_uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Erode mask to shrink inward — this removes the skull ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.erode(mask, kernel, iterations=2)

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