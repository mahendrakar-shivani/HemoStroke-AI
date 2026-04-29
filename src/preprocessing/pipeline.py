# pipeline.py — Combines all preprocessing steps in order

from preprocessing.loader      import load_image
from preprocessing.resize       import resize_image
from preprocessing.normalize    import normalize_image
from preprocessing.noise_removal import remove_noise
from preprocessing.skull_strip  import skull_strip

def preprocess_single(image, target_size=(256, 256),
                       norm_method="minmax",
                       noise_method="gaussian",
                       skull_method="contour"):
    """
    Full preprocessing pipeline for one image.

    Order:
        1. Resize
        2. Normalize
        3. Denoise
        4. Skull strip

    Args:
        image:        raw numpy array from loader
        target_size:  output size (default 256x256)
        norm_method:  'minmax', 'zscore', 'brain_window'
        noise_method: 'gaussian', 'median', 'bilateral'
        skull_method: 'threshold', 'contour'

    Returns:
        clean numpy array ready for model input
    """
    image = resize_image(image, target_size)
    image = normalize_image(image, method=norm_method)
    image = remove_noise(image, method=noise_method)
    image = skull_strip(image, method=skull_method)
    return image

def preprocess_from_path(file_path, **kwargs):
    """
    Load + preprocess in one call.
    Use this in the Streamlit UI.
    """
    image = load_image(file_path)
    return preprocess_single(image, **kwargs)