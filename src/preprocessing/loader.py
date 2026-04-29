# loader.py — Load CT scan images in any format

import cv2
import numpy as np
import pydicom
import nibabel as nib
import os

def load_png(file_path):
    """Load a PNG CT scan as grayscale numpy array."""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {file_path}")
    return image.astype(np.float32)

def load_dicom(file_path):
    """Load a DICOM CT scan and convert to numpy array."""
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept
    return image

def load_nifti(file_path, slice_idx=None):
    """
    Load a NIfTI (.nii or .nii.gz) file.
    slice_idx: which slice to use (default: middle slice)
    """
    nii    = nib.load(file_path)
    volume = nii.get_fdata().astype(np.float32)
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    return volume[:, :, slice_idx]

def load_image(file_path):
    """
    Auto-detect format and load image.
    Supports: .png, .jpg, .dcm, .nii, .nii.gz
    """
    ext = file_path.lower()
    if ext.endswith('.dcm'):
        return load_dicom(file_path)
    elif ext.endswith('.nii') or ext.endswith('.nii.gz'):
        return load_nifti(file_path)
    else:
        return load_png(file_path)