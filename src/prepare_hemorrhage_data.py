# prepare_hemorrhage_data.py — Converts PhysioNet ct-ich NIfTI volumes into
# 2D windowed PNG slices, then merges them with the existing ischemia
# (External_Test) image/mask pairs into one combined segmentation dataset.
#
# After running this, data/Segmentation_Data/ has the same PNG/MASKS/
# labels.csv structure as External_Test — segmentation_dataset.py needs
# NO changes, just point train_segmentation.py's DATA_DIR at the new folder.
#
# Run from repo root:
#   python src/prepare_hemorrhage_data.py

import os
import sys
import shutil

import cv2
import numpy as np
import nibabel as nib
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
ICH_DIR = "data/ct_ich"                    # extracted PhysioNet dataset (ct_scans/, masks/)
EXTERNAL_TEST_DIR = "data/External_Test"    # existing ischemia PNG/MASKS/labels.csv
OUTPUT_DIR = "data/Segmentation_Data"       # combined output

# Brain window: standard clinical setting for distinguishing acute blood
# (hyperdense, ~40-90 HU) from normal brain tissue (~20-40 HU).
WINDOW_LEVEL = 40
WINDOW_WIDTH = 80


def apply_brain_window(slice_hu):
    """Clips raw Hounsfield Units to the brain window, scales to 0-255 uint8."""
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2
    clipped = np.clip(slice_hu, lower, upper)
    scaled = ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return scaled


def convert_ich_volumes(ich_dir, out_png_dir, out_mask_dir):
    """Walks every patient .nii pair, saves one PNG pair per slice."""
    ct_dir = os.path.join(ich_dir, "ct_scans")
    mask_dir = os.path.join(ich_dir, "masks")

    if not os.path.exists(ct_dir):
        print(f"WARNING: {ct_dir} not found — skipping hemorrhage data.")
        return []

    rows = []
    patient_files = sorted(f for f in os.listdir(ct_dir) if f.endswith(".nii"))
    print(f"Converting {len(patient_files)} patient volumes...")

    for fname in patient_files:
        patient_id = os.path.splitext(fname)[0]
        ct_path = os.path.join(ct_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            print(f"  skipping {patient_id}: no matching mask file")
            continue

        ct_vol = nib.load(ct_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()
        n_slices = ct_vol.shape[2]

        for i in range(n_slices):
            image_id = f"ich_{patient_id}_{i:03d}"
            ct_slice = apply_brain_window(ct_vol[:, :, i])
            mask_slice = (mask_vol[:, :, i] > 127).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(out_png_dir, f"{image_id}.png"), ct_slice)
            cv2.imwrite(os.path.join(out_mask_dir, f"{image_id}.png"), mask_slice)

            has_lesion = int(mask_slice.sum() > 0)
            rows.append({"image_id": image_id, "Stroke": has_lesion})

    print(f"  -> {len(rows)} hemorrhage slices written")
    return rows


def copy_ischemia_data(external_test_dir, out_png_dir, out_mask_dir):
    """Copies the existing 200 ischemia PNG/MASK pairs, relabeling Stroke by
    actual mask content (not the original classifier label) so stratified
    splitting reflects real lesion presence."""
    src_png = os.path.join(external_test_dir, "PNG")
    src_mask = os.path.join(external_test_dir, "MASKS")

    if not os.path.exists(src_png):
        print(f"WARNING: {src_png} not found — skipping ischemia data.")
        return []

    rows = []
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(src_png))
    print(f"Copying {len(ids)} ischemia pairs...")

    for image_id in ids:
        src_png_path = os.path.join(src_png, f"{image_id}.png")
        src_mask_path = os.path.join(src_mask, f"{image_id}.png")
        if not os.path.exists(src_mask_path):
            continue

        # re-save as single-channel grayscale to match hemorrhage PNGs exactly
        img = cv2.imread(src_png_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(out_png_dir, f"{image_id}.png"), img)
        cv2.imwrite(os.path.join(out_mask_dir, f"{image_id}.png"), mask)

        has_lesion = int(mask.sum() > 0)
        rows.append({"image_id": image_id, "Stroke": has_lesion})

    print(f"  -> {len(rows)} ischemia pairs copied")
    return rows


def main():
    out_png_dir = os.path.join(OUTPUT_DIR, "PNG")
    out_mask_dir = os.path.join(OUTPUT_DIR, "MASKS")
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    ischemia_rows = copy_ischemia_data(EXTERNAL_TEST_DIR, out_png_dir, out_mask_dir)
    hemorrhage_rows = convert_ich_volumes(ICH_DIR, out_png_dir, out_mask_dir)

    all_rows = ischemia_rows + hemorrhage_rows
    df = pd.DataFrame(all_rows)
    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    df.to_csv(labels_path, index=False)

    n_with_lesion = df["Stroke"].sum()
    print(f"\nCombined dataset: {len(df)} total pairs")
    print(f"  With lesion pixels : {n_with_lesion}")
    print(f"  Empty masks        : {len(df) - n_with_lesion}")
    print(f"Saved labels -> {labels_path}")


if __name__ == "__main__":
    main()
