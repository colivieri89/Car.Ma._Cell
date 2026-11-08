"""Step 3b — skeletonisation of the binary mitochondrial mask."""

from __future__ import annotations

import logging

import numpy as np
from skimage import morphology

logger = logging.getLogger(__name__)


def skeletonize_network(mask: np.ndarray) -> np.ndarray:
    """Skeletonise a binary mask (Lee 1994 algorithm).

    Produces a one-pixel-wide skeleton that preserves topology as much as
    possible.
    """
    logger.info("Skeletonising binary mask (Lee 1994).")
    skeleton = morphology.skeletonize(mask, method="lee")
    return skeleton.astype(bool)
