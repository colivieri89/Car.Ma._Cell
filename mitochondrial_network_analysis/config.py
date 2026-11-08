"""Configuration dataclasses for the mitochondrial network analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class AnalysisConfig:
    """Centralised configuration for the analysis pipeline.

    Attributes
    ----------
    input_path : Path
        Path to the original Step 1 fluorescence image.
    output_dir : Path
        Folder where every output file will be written. Created if missing.
    psf_path : Path, optional
        Optional experimental PSF image. If omitted, a synthetic Gaussian PSF
        is used (controlled by ``psf_size`` and ``psf_sigma``).
    reference_deconvolved_path, reference_skeleton_path : Path, optional
        Optional Fiji/ImageJ reference images. The skeleton reference enables
        Dice/Jaccard/precision/recall validation.
    pixel_size_um : float, optional
        Pixel size in micrometres. If omitted, physical metrics are reported
        as ``None``/``NaN`` unless a scale bar calibration is provided.
    scale_bar_length_px, scale_bar_length_um : float, optional
        Alternative calibration: if both are provided, the effective pixel
        size is ``scale_bar_length_um / scale_bar_length_px``.
    roi : tuple of int, optional
        Optional region of interest as ``(row_start, row_stop, col_start, col_stop)``.
    deconv_iterations, psf_size, psf_sigma : int / int / float
        Richardson-Lucy deconvolution parameters.
    deconv_clip : bool
        If True, clip the deconvolution output to ``[-1, 1]``.
    deconv_filter_epsilon : float, optional
        Optional small value passed to scikit-image's Richardson-Lucy
        implementation for numerical stability.
    threshold_method : str
        One of ``"otsu"``, ``"li"``, ``"yen"``, ``"manual"``.
    manual_threshold : float, optional
        Manual threshold in ``[0, 1]`` (used only when ``threshold_method == "manual"``).
    min_object_size, min_hole_size : int
        Minimum object / hole area in pixels for morphological cleanup.
    apply_opening, apply_closing : bool
        Toggle binary opening/closing during cleanup.
    closing_radius : int
        Disk radius for the binary closing operation.
    save_diagnostic_plot : bool
        Whether to save a 2x3 summary figure.
    verbose : bool
        Enable verbose (DEBUG-level) logging.
    """

    # I/O
    input_path: Path
    output_dir: Path
    psf_path: Optional[Path] = None
    reference_deconvolved_path: Optional[Path] = None
    reference_skeleton_path: Optional[Path] = None

    # Calibration
    pixel_size_um: Optional[float] = None
    scale_bar_length_px: Optional[float] = None
    scale_bar_length_um: Optional[float] = None

    # ROI
    roi: Optional[Tuple[int, int, int, int]] = None

    # Deconvolution
    deconv_iterations: int = 30
    psf_size: int = 21
    psf_sigma: float = 2.0
    deconv_clip: bool = False
    deconv_filter_epsilon: Optional[float] = None

    # Segmentation
    threshold_method: str = "otsu"
    manual_threshold: Optional[float] = None
    min_object_size: int = 20
    min_hole_size: int = 20
    apply_opening: bool = False
    apply_closing: bool = True
    closing_radius: int = 1

    # Misc
    save_diagnostic_plot: bool = True
    verbose: bool = False

    def resolve_pixel_size(self) -> Optional[float]:
        """Return the effective pixel size in micrometres, or ``None``."""
        if self.pixel_size_um is not None and self.pixel_size_um > 0:
            return float(self.pixel_size_um)
        if (
            self.scale_bar_length_px
            and self.scale_bar_length_um
            and self.scale_bar_length_px > 0
        ):
            return float(self.scale_bar_length_um) / float(self.scale_bar_length_px)
        return None

    def __post_init__(self) -> None:
        # Coerce string paths to Path objects (helps when constructed from dicts).
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        if self.psf_path is not None:
            self.psf_path = Path(self.psf_path)
        if self.reference_deconvolved_path is not None:
            self.reference_deconvolved_path = Path(self.reference_deconvolved_path)
        if self.reference_skeleton_path is not None:
            self.reference_skeleton_path = Path(self.reference_skeleton_path)
        method = self.threshold_method.lower()
        if method not in {"otsu", "li", "yen", "manual"}:
            raise ValueError(
                f"Invalid threshold_method='{self.threshold_method}'. "
                "Expected one of: otsu, li, yen, manual."
            )
        self.threshold_method = method
        if method == "manual" and self.manual_threshold is None:
            raise ValueError("manual_threshold must be provided when threshold_method='manual'.")
        if self.deconv_iterations < 1:
            raise ValueError("deconv_iterations must be >= 1.")
        if self.psf_size < 3:
            raise ValueError("psf_size must be >= 3.")
        if self.psf_sigma <= 0:
            raise ValueError("psf_sigma must be > 0.")
        if self.roi is not None:
            roi = tuple(self.roi)
            if len(roi) != 4 or any(not isinstance(v, int) for v in roi):
                raise ValueError(
                    f"ROI must be a tuple of 4 integers (r0, r1, c0, c1); got {self.roi!r}."
                )
            r0, r1, c0, c1 = roi
            if r1 <= r0 or c1 <= c0:
                raise ValueError(f"ROI bounds are inconsistent: {self.roi!r}.")
            self.roi = roi
