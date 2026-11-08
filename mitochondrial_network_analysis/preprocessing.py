"""Step 1 — preprocessing: grayscale conversion, ROI cropping, normalisation."""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float

logger = logging.getLogger(__name__)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a 2-D grayscale ``float`` array."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[-1] == 4:  # drop alpha
            image = image[..., :3]
        if image.shape[-1] == 3:
            gray = rgb2gray(image)
        elif image.shape[-1] == 1:
            gray = image[..., 0]
        else:
            logger.warning(
                "Image has %d channels; using max-intensity projection.", image.shape[-1]
            )
            gray = image.max(axis=-1)
    else:
        raise ValueError(f"Unsupported image dimensionality: shape={image.shape}")
    return img_as_float(gray)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Rescale image intensity to ``[0, 1]`` (float32)."""
    image = img_as_float(image)
    vmin, vmax = float(np.min(image)), float(np.max(image))
    if math.isclose(vmin, vmax):
        logger.warning("Constant image during normalisation; returning zeros.")
        return np.zeros_like(image, dtype=np.float32)
    return rescale_intensity(image, in_range=(vmin, vmax), out_range=(0.0, 1.0)).astype(
        np.float32
    )


def apply_roi(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Crop the image to ``(row_start, row_stop, col_start, col_stop)``.

    Returns the original image if ``roi`` is ``None``.
    """
    if roi is None:
        return image
    r0, r1, c0, c1 = roi
    h, w = image.shape[:2]
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, c0), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid ROI {roi} for image of shape {image.shape}.")
    logger.info("Applying ROI: rows %d:%d, cols %d:%d", r0, r1, c0, c1)
    return image[r0:r1, c0:c1]
