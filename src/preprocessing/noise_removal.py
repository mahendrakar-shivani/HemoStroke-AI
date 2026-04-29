# noise_removal.py — Remove noise from CT scan images

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

def gaussian_denoise(image, sigma=1.0):
    """
    Gaussian blur — smooth general noise.
    sigma: higher = more smoothing (use 0.5-1.5 for CT)
    """
    return gaussian_filter(image, sigma=sigma).astype(np.float32)

def median_denoise(image, size=3):
    """
    Median filter — removes salt-and-pepper noise.
    size: filter window size (3 or 5 recommended)
    """
    return median_filter(image, size=size).astype(np.float32)

def bilateral_denoise(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter — removes noise while preserving edges.
    Best for CT scans as it keeps stroke boundary sharp.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    denoised  = cv2.bilateralFilter(
        img_uint8,
        diameter,
        sigma_color,
        sigma_space
    )
    return (denoised / 255.0).astype(np.float32)

def remove_noise(image, method="gaussian", **kwargs):
    """
    Master noise removal function.
    method: 'gaussian', 'median', or 'bilateral'
    """
    if method == "gaussian":
        return gaussian_denoise(image, **kwargs)
    elif method == "median":
        return median_denoise(image, **kwargs)
    elif method == "bilateral":
        return bilateral_denoise(image, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")