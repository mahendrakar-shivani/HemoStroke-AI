# dataset.py — Loads images, creates labels, handles class imbalance

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CTScanDataset(Dataset):
    """
    Loads CT scan images from:
        data/processed/png/Normal/  → label 0
        data/processed/png/Stroke/  → label 1
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def load_dataset(data_dir):
    """
    Reads all images from Normal/ and Stroke/ subfolders.
    Returns: image_paths list, labels list
    """
    image_paths = []
    labels      = []

    class_map = {"Normal": 0, "Stroke": 1}

    for folder_name, label in class_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder not found → {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for fname in files:
            image_paths.append(os.path.join(folder_path, fname))
            labels.append(label)

    print(f"\nDataset loaded:")
    print(f"  Normal (0) : {labels.count(0)}")
    print(f"  Stroke (1) : {labels.count(1)}")
    print(f"  Total      : {len(labels)}")

    return image_paths, labels


def get_transforms(augment=False):
    """
    augment=True  → used for training   (adds random flips/rotation)
    augment=False → used for testing    (clean, no random changes)
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])