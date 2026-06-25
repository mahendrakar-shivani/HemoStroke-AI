import os
import sys
import cv2
import numpy as np

# Add src/ to path so we can import our preprocessing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.resize import resize_image
from preprocessing.normalize import normalize_image
from preprocessing.noise_removal import remove_noise
from preprocessing.skull_strip import skull_strip

# Base input (raw data)
input_base = "data/raw"

# Output path
output_base = "data/processed/png"

# Create output folders
os.makedirs(os.path.join(output_base, "Normal"), exist_ok=True)
os.makedirs(os.path.join(output_base, "Stroke"), exist_ok=True)


def preprocess(img):
    """
    Full preprocessing pipeline using our tested, fixed modules.
    """
    img = img.astype(np.float32)
    img = resize_image(img, target_size=(256, 256))
    img = normalize_image(img, method="minmax")
    img = remove_noise(img, method="gaussian")
    img = skull_strip(img, method="contour")
    return img


# Mapping
category_map = {
    "Normal": "Normal",
    "Ischemia": "Stroke",
    "Bleeding": "Stroke"
}

count = 0
for folder_name, label in category_map.items():
    folder_path = os.path.join(input_base, folder_name)
    png_folder = os.path.join(folder_path, "PNG")

    if not os.path.exists(png_folder):
        print(f"PNG folder not found: {png_folder}")
        continue

    print(f"\nProcessing folder: {folder_name}")

    for file in os.listdir(png_folder):
        file_path = os.path.join(png_folder, file)

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        processed = preprocess(img)

        save_path = os.path.join(output_base, label, file)
        cv2.imwrite(save_path, (processed * 255).astype(np.uint8))
        count += 1
        print(f"Processed: {file} -> {label}")

print(f"\nDONE! Total images processed: {count}")