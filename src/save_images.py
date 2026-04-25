import numpy as np
import cv2
import os

def save_as_npy(image, save_path):
    """Save preprocessed image as a NumPy array (.npy). Best for training."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, image)
    print(f"Saved as .npy → {save_path}")

def save_as_png(image, save_path):
    """Save preprocessed image as a PNG (for visual verification)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_uint8 = (image * 255).astype(np.uint8)  # scale back to 0–255
    cv2.imwrite(save_path, img_uint8)
    print(f"Saved as .png → {save_path}")

def process_and_save_batch(raw_folder, save_folder, target_size=(256, 256)):
    """
    Process ALL images in a folder and save them.
    Use this to batch-process your entire dataset.
    """
    from load_images import load_png
    from preprocess import preprocess_pipeline

    files = [f for f in os.listdir(raw_folder) if f.endswith('.png')]
    print(f"Found {len(files)} images to process...")

    for i, filename in enumerate(files):
        raw_path  = os.path.join(raw_folder, filename)
        name      = os.path.splitext(filename)[0]
        npy_path  = os.path.join(save_folder, "npy", f"{name}.npy")
        png_path  = os.path.join(save_folder, "png", f"{name}.png")

        try:
            raw = load_png(raw_path).astype(np.float32)
            processed = preprocess_pipeline(raw, target_size)
            save_as_npy(processed, npy_path)
            save_as_png(processed, png_path)
            print(f"[{i+1}/{len(files)}] Done: {filename}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("\nAll images processed and saved!")

if __name__ == "__main__":
    process_and_save_batch(
        raw_folder="data/raw/png/",
        save_folder="data/processed/"
    )