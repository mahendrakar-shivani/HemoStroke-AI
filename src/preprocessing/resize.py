# resize.py — Resize CT scan images to standard size

import cv2
import numpy as np

def resize_image(image, target_size=(256, 256)):
    """
    Resize image to target size.
    Uses INTER_LINEAR for smooth interpolation.
    
    Args:
        image: numpy array (H, W)
        target_size: tuple (width, height) — default 256x256
    Returns:
        resized numpy array
    """
    if image.shape[:2] == target_size:
        return image  # already correct size, skip

    resized = cv2.resize(
        image,
        target_size,
        interpolation=cv2.INTER_LINEAR
    )
    return resized.astype(np.float32)