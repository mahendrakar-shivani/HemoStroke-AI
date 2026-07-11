# segmentation_dataset.py — Loads paired CT scan + lesion mask for Attention-UNet
#
# Expects:
#   <root>/PNG/<id>.png     — CT scan (RGBA or grayscale)
#   <root>/MASKS/<id>.png   — binary lesion mask (0 = background, 255 = lesion)

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.resize import resize_image
from preprocessing.normalize import normalize_minmax


class StrokeSegmentationDataset(Dataset):
    """
    Loads (image, mask) pairs for lesion segmentation.
    image -> float tensor (1, H, W), normalized to [-1, 1]
    mask  -> float tensor (1, H, W), binary (0.0 / 1.0)
    """
    def __init__(self, root_dir, image_ids, target_size=(256, 256), augment=False):
        self.png_dir = os.path.join(root_dir, "PNG")
        self.mask_dir = os.path.join(root_dir, "MASKS")
        self.image_ids = image_ids
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.image_ids)

    def _load_pair(self, image_id):
        img_path = os.path.join(self.png_dir, f"{image_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        # IMREAD_GRAYSCALE collapses RGBA/RGB to single channel automatically
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot load mask: {mask_path}")

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        return image, mask

    def _joint_augment(self, image, mask):
        # Same random flip/rotation applied to BOTH image and mask, so
        # lesion location in the mask stays aligned with the image.
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        k = random.choice([0, 1, 2, 3])  # 0/90/180/270 degree rotation
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        return image, mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, mask = self._load_pair(image_id)

        # image: linear interpolation (smooth). mask: nearest neighbor,
        # so lesion edges stay binary instead of blurring into gray values.
        image = resize_image(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._joint_augment(image, mask)

        image = normalize_minmax(image)          # [0, 1]
        image = (image - 0.5) / 0.5               # [-1, 1], matches CNN pipeline

        mask = (mask > 127).astype(np.float32)     # binarize 0/1

        image_t = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()    # (1, H, W)
        return image_t, mask_t


def load_segmentation_ids(root_dir, val_split=0.2, seed=42):
    """
    Reads labels.csv to get image IDs, then does a stratified split by the
    Stroke column so both train/val sets have a similar lesion-present ratio.
    Returns: train_ids, val_ids
    """
    import pandas as pd

    labels_path = os.path.join(root_dir, "labels.csv")
    df = pd.read_csv(labels_path)
    df["image_id"] = df["image_id"].astype(str)

    # keep only ids that actually have both a PNG and a MASK on disk
    png_dir = os.path.join(root_dir, "PNG")
    mask_dir = os.path.join(root_dir, "MASKS")
    valid_ids = set(os.path.splitext(f)[0] for f in os.listdir(png_dir))
    valid_ids &= set(os.path.splitext(f)[0] for f in os.listdir(mask_dir))
    df = df[df["image_id"].isin(valid_ids)]

    rng = random.Random(seed)
    train_ids, val_ids = [], []
    for label in df["Stroke"].unique():
        ids = df[df["Stroke"] == label]["image_id"].tolist()
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_split))
        val_ids += ids[:n_val]
        train_ids += ids[n_val:]

    print(f"\nSegmentation dataset loaded:")
    print(f"  Train : {len(train_ids)}")
    print(f"  Val   : {len(val_ids)}")

    return train_ids, val_ids
