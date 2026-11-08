"""Validation against an external (Fiji/ImageJ) reference skeleton."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from skimage.transform import resize
from skimage.util import img_as_float

from .io import load_image
from .preprocessing import convert_to_grayscale

logger = logging.getLogger(__name__)


def _binarise_reference(image: np.ndarray) -> np.ndarray:
    """Convert a reference skeleton image to a strict boolean mask."""
    if image.ndim == 3:
        image = convert_to_grayscale(image)
    if image.dtype == bool:
        return image
    arr = img_as_float(image)
    threshold = 0.5 * float(arr.max()) if arr.max() > 0 else 0.0
    return arr > threshold


def compare_with_reference_skeleton(
    python_skeleton: np.ndarray, reference_path: Path
) -> Dict[str, float]:
    """Compute Dice, Jaccard, precision and recall vs an external reference."""
    reference_path = Path(reference_path)
    ref = _binarise_reference(load_image(reference_path))
    if ref.shape != python_skeleton.shape:
        logger.warning(
            "Reference shape %s != Python shape %s; resizing reference (nearest neighbour).",
            ref.shape,
            python_skeleton.shape,
        )
        ref = (
            resize(ref.astype(float), python_skeleton.shape, order=0, preserve_range=True)
            > 0.5
        )

    a = python_skeleton.astype(bool)
    b = ref.astype(bool)
    intersection = float(np.logical_and(a, b).sum())
    a_sum, b_sum = float(a.sum()), float(b.sum())
    union = float(np.logical_or(a, b).sum())

    return {
        "dice_coefficient": (2.0 * intersection) / (a_sum + b_sum) if (a_sum + b_sum) > 0 else float("nan"),
        "jaccard_index": intersection / union if union > 0 else float("nan"),
        "precision": intersection / a_sum if a_sum > 0 else float("nan"),
        "recall": intersection / b_sum if b_sum > 0 else float("nan"),
        "python_skeleton_pixels": int(a_sum),
        "reference_skeleton_pixels": int(b_sum),
        "intersection_pixels": int(intersection),
    }
