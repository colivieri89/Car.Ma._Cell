"""Step 3a — thresholding and morphological cleanup."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from skimage import filters, morphology

logger = logging.getLogger(__name__)


def remove_small_objects_compat(mask: np.ndarray, min_size: int) -> np.ndarray:
    """``morphology.remove_small_objects`` that works across skimage versions.

    scikit-image 0.26 renamed the ``min_size`` parameter to ``max_size``. We
    pass the same numeric value and prefer the new name, falling back to the old
    one on older skimage. The exact size boundary (inclusive vs. exclusive)
    therefore follows the installed skimage version -- identical to a direct
    ``remove_small_objects`` call on that version, just without the deprecation
    warning on 0.26+.
    """
    try:
        return morphology.remove_small_objects(mask, max_size=min_size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=min_size)


def remove_small_holes_compat(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """``morphology.remove_small_holes`` that works across skimage versions.

    scikit-image 0.26 renamed ``area_threshold`` to ``max_size`` (same numeric
    value passed); the size boundary follows the installed skimage version.
    """
    try:
        return morphology.remove_small_holes(mask, max_size=area_threshold)
    except TypeError:
        return morphology.remove_small_holes(mask, area_threshold=area_threshold)


def _compute_threshold(
    image: np.ndarray, method: str, manual: Optional[float]
) -> float:
    method = method.lower()
    if method == "otsu":
        return float(filters.threshold_otsu(image))
    if method == "li":
        return float(filters.threshold_li(image))
    if method == "yen":
        return float(filters.threshold_yen(image))
    if method == "manual":
        if manual is None:
            raise ValueError("manual_threshold must be provided when method='manual'.")
        return float(manual)
    raise ValueError(f"Unknown threshold method: {method}")


def create_binary_mask(
    image: np.ndarray, method: str = "otsu", manual_threshold: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Return ``(mask, threshold)`` from a (deconvolved) intensity image."""
    threshold = _compute_threshold(image, method, manual_threshold)
    logger.info("Thresholding (method=%s, threshold=%.4f)", method, threshold)
    return image > threshold, threshold


def clean_binary_mask(
    mask: np.ndarray,
    min_object_size: int = 20,
    min_hole_size: int = 20,
    apply_opening: bool = False,
    apply_closing: bool = True,
    closing_radius: int = 1,
) -> np.ndarray:
    """Apply morphological cleanup to a binary mask."""
    cleaned = mask.astype(bool, copy=True)
    if apply_opening:
        cleaned = morphology.opening(cleaned, footprint=morphology.disk(1))
    if apply_closing and closing_radius > 0:
        cleaned = morphology.closing(
            cleaned, footprint=morphology.disk(closing_radius)
        )
    if min_hole_size > 0:
        cleaned = remove_small_holes_compat(cleaned, min_hole_size)
    if min_object_size > 0:
        cleaned = remove_small_objects_compat(cleaned, min_object_size)
    return cleaned
