"""End-to-end pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd

from .config import AnalysisConfig
from .deconvolution import generate_gaussian_psf, load_psf, run_deconvolution
from .graph_analysis import build_skeleton_graph, extract_branches
from .io import load_image, save_image
from .metrics import (
    compute_branch_metrics,
    compute_component_metrics,
    compute_global_metrics,
    export_metrics,
)
from .preprocessing import apply_roi, convert_to_grayscale, normalize_image
from .segmentation import clean_binary_mask, create_binary_mask
from .skeletonization import skeletonize_network
from .validation import compare_with_reference_skeleton
from .visualization import (
    create_overlay,
    save_diagnostic_plot,
    save_metrics_report,
    save_metrics_text_report,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: AnalysisConfig) -> Dict[str, object]:
    """Execute the full mitochondrial network analysis pipeline.

    Parameters
    ----------
    config : AnalysisConfig
        All inputs, outputs and processing parameters.

    Returns
    -------
    dict
        Summary with the resolved configuration, the global metrics, and
        (optionally) the validation metrics.
    """
    logger.info("=== Mitochondrial network analysis pipeline ===")
    logger.info("Input : %s", config.input_path)
    logger.info("Output: %s", config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size_um = config.resolve_pixel_size()
    if pixel_size_um:
        logger.info("Effective pixel size: %.5f um/px", pixel_size_um)
    else:
        logger.info("No pixel size provided - physical metrics will be None.")

    # ---- Step 1 -------------------------------------------------------
    raw = load_image(config.input_path)
    gray = convert_to_grayscale(raw)
    gray = apply_roi(gray, config.roi)
    original_norm = normalize_image(gray)
    save_image(original_norm, config.output_dir / "step1_original_normalized.tif")

    # ---- Step 2 -------------------------------------------------------
    if config.psf_path is not None:
        psf = load_psf(config.psf_path)
        logger.info("Loaded experimental PSF (%s, shape=%s).", config.psf_path, psf.shape)
    else:
        psf = generate_gaussian_psf(config.psf_size, config.psf_sigma)
        logger.info(
            "Generated synthetic Gaussian PSF (size=%d, sigma=%.2f).",
            config.psf_size,
            config.psf_sigma,
        )

    deconvolved = run_deconvolution(
        original_norm,
        psf,
        iterations=config.deconv_iterations,
        clip=config.deconv_clip,
        filter_epsilon=config.deconv_filter_epsilon,
    )
    save_image(deconvolved, config.output_dir / "step2_deconvolved.tif")

    # ---- Step 3 -------------------------------------------------------
    mask, threshold = create_binary_mask(
        deconvolved,
        method=config.threshold_method,
        manual_threshold=config.manual_threshold,
    )
    mask = clean_binary_mask(
        mask,
        min_object_size=config.min_object_size,
        min_hole_size=config.min_hole_size,
        apply_opening=config.apply_opening,
        apply_closing=config.apply_closing,
        closing_radius=config.closing_radius,
    )
    save_image(mask, config.output_dir / "binary_mask.tif")

    skeleton = skeletonize_network(mask)
    save_image(skeleton, config.output_dir / "step3_skeleton.tif")

    create_overlay(
        deconvolved, skeleton, config.output_dir / "overlay_skeleton_on_deconvolved.png"
    )
    create_overlay(
        original_norm, skeleton, config.output_dir / "overlay_skeleton_on_original.png"
    )

    # ---- Network analysis --------------------------------------------
    graph = build_skeleton_graph(skeleton)
    branches, components_sorted = extract_branches(graph)

    roi_area_px = int(original_norm.shape[0] * original_norm.shape[1])
    global_metrics = compute_global_metrics(
        skeleton, graph, branches, components_sorted, pixel_size_um, roi_area_px,
        mask=mask,
    )
    global_metrics["threshold_value"] = float(threshold)
    global_metrics["threshold_method"] = config.threshold_method

    component_df = compute_component_metrics(
        graph, branches, components_sorted, pixel_size_um
    )
    branch_df = compute_branch_metrics(
        branches, graph, components_sorted, pixel_size_um
    )

    export_metrics(config.output_dir, global_metrics, component_df, branch_df)

    if config.save_diagnostic_plot:
        save_diagnostic_plot(
            original_norm,
            deconvolved,
            mask,
            skeleton,
            config.output_dir / "diagnostic_plot.png",
        )

    # ---- Optional validation -----------------------------------------
    validation: Optional[Dict[str, float]] = None
    if config.reference_skeleton_path is not None:
        logger.info(
            "Validating against reference skeleton: %s", config.reference_skeleton_path
        )
        validation = compare_with_reference_skeleton(
            skeleton, config.reference_skeleton_path
        )
        pd.DataFrame([validation]).to_csv(
            config.output_dir / "validation_metrics.csv", index=False
        )
        logger.info("Validation metrics: %s", validation)

    # ---- Step 4 - Metrics output panel --------------------------------
    save_metrics_report(
        global_metrics,
        branch_df,
        component_df,
        config.output_dir / "step4_metrics_report.png",
        validation=validation,
    )
    save_metrics_text_report(
        global_metrics,
        config.output_dir / "step4_metrics_summary.txt",
        validation=validation,
    )

    logger.info("=== Pipeline complete ===")
    return {
        "config": asdict(config),
        "global_metrics": global_metrics,
        "n_components": int(global_metrics["n_connected_components"]),
        "n_branches": int(global_metrics["n_branches"]),
        "validation": validation,
    }
