# show_preprocessing.py — Show before and after preprocessing for any CT scan

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.loader       import load_image
from preprocessing.resize       import resize_image
from preprocessing.normalize    import normalize_image
from preprocessing.noise_removal import remove_noise
from preprocessing.skull_strip  import skull_strip

def show_preprocessing_steps(image_path, save_path="outputs/plots/preprocessing_steps.png"):
    """
    Shows the CT scan at every stage of preprocessing.
    Creates a visual comparison: Original → Each Step → Final
    """
    os.makedirs("outputs/plots", exist_ok=True)

    # ── Load original ─────────────────────────────────────────
    original = load_image(image_path)

    # ── Apply each step separately ────────────────────────────
    step1_resized    = resize_image(original)
    step2_normalized = normalize_image(step1_resized, method="minmax")
    step3_denoised   = remove_noise(step2_normalized, method="gaussian")
    step4_final      = skull_strip(step3_denoised, method="contour")

    # ── Plot all 5 stages ─────────────────────────────────────
    titles = [
        f"1. Original\n{original.shape[0]}×{original.shape[1]} px",
        f"2. Resized\n256×256 px",
        f"3. Normalized\n[0, 1] range",
        f"4. Denoised\nGaussian filter",
        f"5. Skull Stripped\nBrain only"
    ]

    images = [
        original,
        step1_resized,
        step2_normalized,
        step3_denoised,
        step4_final
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("HemoStroke-AI — CT Scan Preprocessing Pipeline",
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis('off')

        # Show pixel value range below each image
        ax.set_xlabel(f"Min: {img.min():.2f}  Max: {img.max():.2f}",
                      fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


def show_before_after(image_path, save_path="outputs/plots/before_after.png"):
    """
    Simple 2-image comparison: Original vs Final processed.
    Clean version for presentation slides.
    """
    os.makedirs("outputs/plots", exist_ok=True)

    # Load and process
    original  = load_image(image_path)
    processed = resize_image(original)
    processed = normalize_image(processed)
    processed = remove_noise(processed)
    processed = skull_strip(processed)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("HemoStroke-AI — Before vs After Preprocessing",
                 fontsize=13, fontweight='bold')

    ax1.imshow(original, cmap='gray')
    ax1.set_title("BEFORE\nRaw CT Scan", fontsize=12, color='tomato', fontweight='bold')
    ax1.set_xlabel(f"Size: {original.shape[0]}×{original.shape[1]}\n"
                   f"Range: [{original.min():.0f}, {original.max():.0f}]",
                   fontsize=9)
    ax1.axis('off')

    ax2.imshow(processed, cmap='gray')
    ax2.set_title("AFTER\nPreprocessed CT Scan", fontsize=12, color='steelblue', fontweight='bold')
    ax2.set_xlabel(f"Size: 256×256\n"
                   f"Range: [0.00, 1.00]",
                   fontsize=9)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")


# ── Run both visualizations ───────────────────────────────────────────────────
if __name__ == "__main__":

    # Find any image from your processed folder
    search_folders = [
        "data/processed/png/Normal",
        "data/processed/png/Stroke",
        "data/raw/Normal/png",
    ]

    image_path = None
    for folder in search_folders:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.png')]
            if files:
                image_path = os.path.join(folder, files[0])
                break

    if image_path is None:
        print("No image found. Set image_path manually.")
    else:
        print(f"Using image: {image_path}")

        # 1. Simple before/after (best for presentation)
        show_before_after(image_path)

        # 2. All 5 steps shown
        show_preprocessing_steps(image_path)