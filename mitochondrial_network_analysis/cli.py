"""Command-line interface for the mitochondrial network analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from ._deps import ensure_dependencies
from .config import AnalysisConfig

logger = logging.getLogger("mitochondrial_network_analysis")


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mito-analyze",
        description=(
            "Mitochondrial network analysis pipeline (Python equivalent of an "
            "ImageJ/Fiji deconvolution + skeletonisation workflow)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the Step 1 fluorescence image.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for all results.")
    parser.add_argument("--psf", type=Path, default=None,
                        help="Optional experimental PSF image.")
    parser.add_argument("--reference-deconvolved", type=Path, default=None,
                        help="Optional Fiji deconvolved reference image.")
    parser.add_argument("--reference-skeleton", type=Path, default=None,
                        help="Optional Fiji skeleton reference image (enables validation).")

    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--scale-bar-length-px", type=float, default=None)
    parser.add_argument("--scale-bar-length-um", type=float, default=None)
    parser.add_argument("--roi", type=int, nargs=4, default=None,
                        metavar=("R0", "R1", "C0", "C1"),
                        help="Optional ROI: row_start row_stop col_start col_stop.")

    parser.add_argument("--deconv-iterations", type=int, default=30)
    parser.add_argument("--psf-size", type=int, default=21)
    parser.add_argument("--psf-sigma", type=float, default=2.0)
    parser.add_argument("--deconv-clip", action="store_true")
    parser.add_argument("--deconv-filter-epsilon", type=float, default=None)

    parser.add_argument("--threshold-method", choices=["otsu", "li", "yen", "manual"],
                        default="otsu")
    parser.add_argument("--manual-threshold", type=float, default=None)
    parser.add_argument("--min-object-size", type=int, default=20)
    parser.add_argument("--min-hole-size", type=int, default=20)
    parser.add_argument("--apply-opening", action="store_true")
    parser.add_argument("--no-closing", action="store_true",
                        help="Disable binary closing.")
    parser.add_argument("--closing-radius", type=int, default=1)

    parser.add_argument("--no-diagnostic-plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        input_path=args.input,
        output_dir=args.output,
        psf_path=args.psf,
        reference_deconvolved_path=args.reference_deconvolved,
        reference_skeleton_path=args.reference_skeleton,
        pixel_size_um=args.pixel_size_um,
        scale_bar_length_px=args.scale_bar_length_px,
        scale_bar_length_um=args.scale_bar_length_um,
        roi=tuple(args.roi) if args.roi is not None else None,
        deconv_iterations=args.deconv_iterations,
        psf_size=args.psf_size,
        psf_sigma=args.psf_sigma,
        deconv_clip=args.deconv_clip,
        deconv_filter_epsilon=args.deconv_filter_epsilon,
        threshold_method=args.threshold_method,
        manual_threshold=args.manual_threshold,
        min_object_size=args.min_object_size,
        min_hole_size=args.min_hole_size,
        apply_opening=args.apply_opening,
        apply_closing=not args.no_closing,
        closing_radius=args.closing_radius,
        save_diagnostic_plot=not args.no_diagnostic_plot,
        verbose=args.verbose,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m mitochondrial_network_analysis.cli``."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        ensure_dependencies()
    except ImportError as exc:
        logger.error("%s", exc)
        return 2

    # Import the pipeline after the dependency check so we get clean errors.
    from .pipeline import run_pipeline

    try:
        config = _config_from_args(args)
        run_pipeline(config)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
