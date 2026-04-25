import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# STEP 4A: Resize to a standard size
# ─────────────────────────────────────────────
def resize_image(image, target_size=(256, 256)):
    """Resize CT scan to a fixed size for the model."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    print(f"Resized: {image.shape} → {resized.shape}")
    return resized


# ─────────────────────────────────────────────
# STEP 4B: Normalize pixel values to [0, 1]
# ─────────────────────────────────────────────
def normalize_image(image):
    """Scale pixel values to 0–1 range. Required before feeding to neural networks."""
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)  # avoid division by zero
    
    normalized = (image - img_min) / (img_max - img_min)
    print(f"Normalized: Min={normalized.min():.3f} | Max={normalized.max():.3f}")
    return normalized.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 4C: Noise removal (Gaussian blur)
# ─────────────────────────────────────────────
def remove_noise(image, sigma=1.0):
    """Apply Gaussian blur to reduce scanner noise."""
    denoised = gaussian_filter(image, sigma=sigma)
    print(f"Noise removed with sigma={sigma}")
    return denoised


# ─────────────────────────────────────────────
# STEP 4D: Skull stripping (remove skull, keep brain)
# ─────────────────────────────────────────────
def skull_strip(image):
    """
    Simple skull stripping using thresholding + morphology.
    For real projects, use FSL BET or HD-BET for better accuracy.
    """
    # Convert to uint8 for OpenCV morphology
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Threshold: keep only brain tissue values
    _, binary = cv2.threshold(img_uint8, 10, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to image
    stripped = image * (mask / 255.0)
    print("Skull stripping done.")
    return stripped


# ─────────────────────────────────────────────
# FULL PIPELINE: Run all steps in order
# ─────────────────────────────────────────────
def preprocess_pipeline(image, target_size=(256, 256)):
    """
    Complete preprocessing pipeline.
    Input:  raw CT image (numpy array)
    Output: clean, normalized image ready for the model
    """
    print("\n--- Starting Preprocessing ---")
    image = resize_image(image, target_size)    # Step 1: resize
    image = normalize_image(image)              # Step 2: normalize
    image = remove_noise(image, sigma=0.8)      # Step 3: denoise
    image = skull_strip(image)                  # Step 4: skull strip
    print("--- Preprocessing Complete ---\n")
    return image


# ─────────────────────────────────────────────
# TEST: Quick visual check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from load_images import load_png

    raw = load_png("data/raw/png/sample.png")
    processed = preprocess_pipeline(raw.astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw, cmap='gray');       axes[0].set_title("Original")
    axes[1].imshow(processed, cmap='gray'); axes[1].set_title("Preprocessed")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/preprocessing_comparison.png", dpi=150)
    plt.show()