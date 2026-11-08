"""Step 2 — Richardson–Lucy deconvolution with synthetic or experimental PSF.

The numerical result is not bit-identical to Fiji/ImageJ. The Fiji output
depends on the chosen plugin (e.g. ``DeconvolutionLab2``,
``Iterative Deconvolve 3D``), the exact PSF and microscope parameters. This
module implements a scientifically equivalent workflow using
``skimage.restoration.richardson_lucy``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from skimage import restoration
from skimage.util import img_as_float

from .io import load_image
from .preprocessing import convert_to_grayscale, normalize_image

logger = logging.getLogger(__name__)


def generate_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Generate a normalised 2-D Gaussian point-spread function."""
    if size < 3:
        raise ValueError("PSF size must be >= 3.")
    if sigma <= 0:
        raise ValueError("PSF sigma must be > 0.")
    if size % 2 == 0:
        size += 1  # force odd size for centring
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf.astype(np.float32)


def generate_theoretical_psf(
    numerical_aperture: float,
    wavelength_em_nm: float,
    pixel_size_nm: float,
) -> np.ndarray:
    """Generate a 2-D Gaussian PSF from optical parameters.

    Uses the Rayleigh-Gaussian approximation
    ``sigma_lateral = 0.21 * lambda_em / NA`` (Zhang, Zerubia & Olivo-Marin
    2007), converted to pixels through ``pixel_size_nm``. The kernel radius
    is set to ``ceil(4 * sigma_px)`` and the PSF is L1-normalised.

    Adapted from ``MitoNet._generate_psf_theoretical``.
    """
    if numerical_aperture <= 0:
        raise ValueError("Numerical aperture must be > 0.")
    if wavelength_em_nm <= 0:
        raise ValueError("Emission wavelength must be > 0 nm.")
    if pixel_size_nm <= 0:
        raise ValueError("Pixel size must be > 0 nm.")

    sigma_px = (0.21 * wavelength_em_nm / numerical_aperture) / pixel_size_nm
    if sigma_px <= 0:
        raise ValueError("Computed PSF sigma is non-positive.")
    radius = int(np.ceil(4.0 * sigma_px))
    if radius < 1:
        radius = 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    psf = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma_px ** 2))
    psf = psf / psf.sum()
    logger.info(
        "Theoretical PSF: NA=%.3f, lambda_em=%.1f nm, pixel=%.1f nm -> "
        "sigma=%.3f px, kernel=%dx%d",
        numerical_aperture,
        wavelength_em_nm,
        pixel_size_nm,
        sigma_px,
        psf.shape[0],
        psf.shape[1],
    )
    return psf.astype(np.float32)


def load_psf(path: Path) -> np.ndarray:
    """Load and normalise an experimental PSF image."""
    psf = convert_to_grayscale(load_image(Path(path))).astype(np.float32)
    total = float(psf.sum())
    if total <= 0:
        raise ValueError(f"Loaded PSF has non-positive sum: {total}")
    return psf / total


def run_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 30,
    clip: bool = False,
    filter_epsilon: Optional[float] = None,
) -> np.ndarray:
    """Run Richardson-Lucy deconvolution and return a normalised image."""
    logger.info(
        "Richardson-Lucy deconvolution: iterations=%d, psf_shape=%s",
        iterations,
        psf.shape,
    )
    image = img_as_float(image).astype(np.float32)
    kwargs = {"num_iter": iterations, "clip": clip}
    if filter_epsilon is not None:
        kwargs["filter_epsilon"] = filter_epsilon
    try:
        deconv = restoration.richardson_lucy(image, psf, **kwargs)
    except TypeError:
        # Older scikit-image: parameter name was ``iterations``.
        kwargs.pop("num_iter")
        kwargs["iterations"] = iterations
        deconv = restoration.richardson_lucy(image, psf, **kwargs)
    deconv = np.nan_to_num(deconv, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_image(deconv)
