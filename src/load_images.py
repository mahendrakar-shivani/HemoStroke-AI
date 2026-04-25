import cv2
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# LOADER 1: PNG images (simplest format)
# ─────────────────────────────────────────────
def load_png(file_path):
    """Load a PNG CT scan image using OpenCV."""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # grayscale = 1 channel
    if image is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    print(f"PNG loaded | Shape: {image.shape} | Dtype: {image.dtype}")
    return image


# ─────────────────────────────────────────────
# LOADER 2: DICOM images (hospital standard)
# ─────────────────────────────────────────────
def load_dicom(file_path):
    """Load a DICOM CT scan file."""
    ds = pydicom.dcmread(file_path)         # read the DICOM file
    image = ds.pixel_array                  # extract the pixel data as numpy array
    
    # Apply DICOM rescale values if present (converts raw values to Hounsfield Units)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept

    print(f"DICOM loaded | Shape: {image.shape} | Min: {image.min()} | Max: {image.max()}")
    return image.astype(np.float32)


# ─────────────────────────────────────────────
# LOADER 3: NIfTI images (research standard)
# ─────────────────────────────────────────────
def load_nifti(file_path):
    """Load a NIfTI (.nii or .nii.gz) file and return all slices."""
    nii = nib.load(file_path)
    volume = nii.get_fdata()                # shape: (width, height, num_slices)
    print(f"NIfTI loaded | Volume shape: {volume.shape} | Num slices: {volume.shape[2]}")
    return volume


# ─────────────────────────────────────────────
# TEST: Run this to verify your loaders work
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test PNG
    png_path = "data/raw/png/sample.png"
    if os.path.exists(png_path):
        img = load_png(png_path)
        plt.imshow(img, cmap='gray')
        plt.title("Sample CT Scan (PNG)")
        plt.axis('off')
        plt.show()

    # Test DICOM
    dcm_path = "data/raw/dicom/sample.dcm"
    if os.path.exists(dcm_path):
        img = load_dicom(dcm_path)
        plt.imshow(img, cmap='gray')
        plt.title("Sample CT Scan (DICOM)")
        plt.axis('off')
        plt.show()