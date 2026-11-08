"""Image input/output helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage import io
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk (TIFF / PNG / JPG)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    logger.info("Loading image: %s", path)
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(str(path))
    return io.imread(str(path))


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an image to disk choosing the format from the file extension.

    - TIFF preserves floating-point precision (saved as float32).
    - PNG/JPG converts floats in [0, 1] to uint8.
    - Boolean masks are saved as uint8 ``0/255``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        if image.dtype == bool:
            tifffile.imwrite(str(path), (image.astype(np.uint8) * 255))
        elif np.issubdtype(image.dtype, np.floating):
            tifffile.imwrite(str(path), image.astype(np.float32))
        else:
            tifffile.imwrite(str(path), image)
    else:
        if image.dtype == bool:
            io.imsave(str(path), image.astype(np.uint8) * 255, check_contrast=False)
        elif np.issubdtype(image.dtype, np.floating):
            io.imsave(
                str(path), img_as_ubyte(np.clip(image, 0.0, 1.0)), check_contrast=False
            )
        else:
            io.imsave(str(path), image, check_contrast=False)
    logger.debug("Saved %s (shape=%s, dtype=%s)", path, image.shape, image.dtype)
