import os
import cv2
import numpy as np

# Base input (raw data)
input_base = "data/raw"

# Output path
output_base = "data/processed/png"

# Create output folders
os.makedirs(os.path.join(output_base, "Normal"), exist_ok=True)
os.makedirs(os.path.join(output_base, "Stroke"), exist_ok=True)

# Preprocessing function
def preprocess(img):
    # Resize
    img = cv2.resize(img, (256, 256))

    # Normalize
    img = img / 255.0

    # Noise removal
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Skull stripping (simple threshold)
    _, mask = cv2.threshold(img, 0.1, 1.0, cv2.THRESH_BINARY)
    img = img * mask

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

    # Go inside PNG folder
    png_folder = os.path.join(folder_path, "PNG")

    if not os.path.exists(png_folder):
        print(f"❌ PNG folder not found: {png_folder}")
        continue

    print(f"\n📂 Processing folder: {folder_name}")

    for file in os.listdir(png_folder):
        file_path = os.path.join(png_folder, file)

        # Read image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Preprocess
        processed = preprocess(img)

        # Save path
        save_path = os.path.join(output_base, label, file)

        # Save image
        cv2.imwrite(save_path, (processed * 255).astype(np.uint8))

        count += 1
        print(f"✔ Processed: {file} → {label}")

print(f"\n✅ DONE! Total images processed: {count}")