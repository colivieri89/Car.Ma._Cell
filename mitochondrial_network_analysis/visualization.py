"""Visualisation helpers: overlays, diagnostic plots and Step-4 metrics report."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .preprocessing import normalize_image  # noqa: E402

logger = logging.getLogger(__name__)


def create_overlay(background: np.ndarray, skeleton: np.ndarray, path: Path) -> None:
    """Save a PNG overlay of the skeleton (red) on a grayscale background."""
    bg = normalize_image(background)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(bg, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*skeleton.shape, 4), dtype=np.float32)
    overlay[skeleton, 0] = 1.0  # red channel
    overlay[skeleton, 3] = 1.0  # alpha
    ax.imshow(overlay, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    logger.debug("Overlay written: %s", path)


def save_diagnostic_plot(
    original: np.ndarray,
    deconvolved: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    path: Path,
) -> None:
    """Save a 2x3 diagnostic figure summarising the pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Step 1 - Original (normalised)")
    axes[0, 1].imshow(deconvolved, cmap="gray")
    axes[0, 1].set_title("Step 2 - Deconvolved")
    axes[0, 2].imshow(mask, cmap="gray")
    axes[0, 2].set_title("Binary mask")
    axes[1, 0].imshow(skeleton, cmap="gray")
    axes[1, 0].set_title("Step 3 - Skeleton")
    axes[1, 1].imshow(deconvolved, cmap="gray")
    axes[1, 1].imshow(
        np.ma.masked_where(~skeleton, skeleton), cmap="autumn", interpolation="nearest"
    )
    axes[1, 1].set_title("Skeleton on deconvolved")
    axes[1, 2].imshow(original, cmap="gray")
    axes[1, 2].imshow(
        np.ma.masked_where(~skeleton, skeleton), cmap="autumn", interpolation="nearest"
    )
    axes[1, 2].set_title("Skeleton on original")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Diagnostic plot saved: %s", path)


# ---------------------------------------------------------------------------
# Step 4 - Metrics report (standalone "tab")
# ---------------------------------------------------------------------------


# Logical groups used in the Step-4 report. Each entry: display label -> metric key.
_METRIC_GROUPS = (
    (
        "Topology",
        (
            ("Skeleton pixels",            "skeleton_pixels"),
            ("Endpoints",                  "n_endpoints"),
            ("Junction nodes",             "n_junction_nodes"),
            ("Branches / filaments",       "n_branches"),
            ("Connected components",       "n_connected_components"),
            ("Isolated components",        "n_isolated_components"),
            ("Isolated filaments",         "n_isolated_filaments"),
            ("Non-isolated filaments",     "n_non_isolated_filaments"),
        ),
    ),
    (
        "Lengths",
        (
            ("Total skeleton length [px]", "total_skeleton_length_px"),
            ("Total skeleton length [um]", "total_skeleton_length_um"),
            ("Mean branch length [px]",    "mean_branch_length_px"),
            ("Median branch length [px]",  "median_branch_length_px"),
            ("Min branch length [px]",     "min_branch_length_px"),
            ("Max branch length [px]",     "max_branch_length_px"),
            ("Std branch length [px]",     "std_branch_length_px"),
            ("Mean branch length [um]",    "mean_branch_length_um"),
        ),
    ),
    (
        "Cell normalisation",
        (
            ("Pixel size [um/px]",         "pixel_size_um"),
            ("ROI area [px]",              "roi_area_px"),
            ("Cell area (hull) [px]",      "cell_area_px"),
            ("Cell area (hull) [um^2]",    "cell_area_um2"),
            ("Mitochondrial area [px]",    "mitochondrial_area_px"),
            ("Mito / cell area fraction",  "mitochondrial_area_fraction"),
            ("Network density [1/px]",     "network_density_px_per_px2"),
            ("Network density [1/um]",     "network_density_um_per_um2"),
        ),
    ),
    (
        "Branching",
        (
            ("Branching index [1/px^2]",   "branching_index_per_px2"),
            ("Branching index [1/um^2]",   "branching_index_per_um2"),
            ("Branches density [1/px^2]",  "branches_per_px2"),
            ("Branches density [1/um^2]",  "branches_per_um2"),
            ("Junctions / unit length",    "junctions_per_unit_length_px"),
            ("Endpoints / unit length",    "endpoints_per_unit_length_px"),
            ("Normalised skeleton length", "normalised_skeleton_length"),
        ),
    ),
    (
        "Connectivity & Fragmentation",
        (
            ("Connectivity index (length)", "connectivity_index_length"),
            ("Connectivity index (count)",  "connectivity_index_count"),
            ("Fragmentation index (length)", "fragmentation_index_length"),
            ("Fragmentation index (count)",  "fragmentation_index_count"),
            ("Components density [1/px^2]",  "components_per_px2"),
            ("Components density [1/um^2]",  "components_per_um2"),
        ),
    ),
    (
        "Complexity",
        (
            ("Network complexity index",    "network_complexity_index"),
        ),
    ),
)


def _format_value(v: object) -> str:
    """Pretty-print metric values (handles None / NaN / floats)."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, float) or isinstance(v, np.floating):
        f = float(v)
        if np.isnan(f):
            return "n/a"
        if f == 0.0:
            return "0"
        if abs(f) >= 1000 or (abs(f) < 1e-3 and f != 0.0):
            return f"{f:.3e}"
        return f"{f:.4f}"
    return str(v)


def save_metrics_report(
    global_metrics: Dict[str, object],
    branch_df: pd.DataFrame,
    component_df: pd.DataFrame,
    path: Path,
    validation: Optional[Dict[str, float]] = None,
) -> None:
    """Save the Step 4 figure: a standalone metrics output panel.

    The figure contains:

    * a tabular text panel grouping every global metric by category;
    * a histogram of branch length (px);
    * a bar chart of the top-10 components by skeleton length;
    * an optional validation summary (Dice / Jaccard / precision / recall).
    """
    fig = plt.figure(figsize=(16, 11), dpi=140)
    fig.suptitle(
        "Step 4 - Mitochondrial Network Metrics",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.4, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.25,
    )

    # ---- Left panel: grouped metrics table (text) ----------------------
    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.set_axis_off()

    lines: list[str] = []
    for group_name, items in _METRIC_GROUPS:
        lines.append(f"[{group_name}]")
        for label, key in items:
            value = global_metrics.get(key)
            lines.append(f"  {label:<32s} : {_format_value(value)}")
        lines.append("")

    if validation:
        lines.append("[Validation vs reference skeleton]")
        for label, key in (
            ("Dice coefficient",          "dice_coefficient"),
            ("Jaccard index",             "jaccard_index"),
            ("Precision",                 "precision"),
            ("Recall",                    "recall"),
            ("Python skeleton pixels",    "python_skeleton_pixels"),
            ("Reference skeleton pixels", "reference_skeleton_pixels"),
            ("Intersection pixels",       "intersection_pixels"),
        ):
            lines.append(f"  {label:<32s} : {_format_value(validation.get(key))}")

    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        family="monospace",
        fontsize=9.5,
        va="top",
        ha="left",
        transform=ax_text.transAxes,
    )

    # ---- Top-right: branch-length histogram ---------------------------
    ax_hist = fig.add_subplot(gs[0, 1])
    if not branch_df.empty and "length_px" in branch_df.columns:
        lens = branch_df.loc[branch_df["n_pixels"] > 1, "length_px"].to_numpy(dtype=float)
        if lens.size:
            ax_hist.hist(lens, bins=min(40, max(8, lens.size // 4)), color="#4C72B0",
                         edgecolor="black", alpha=0.85)
            ax_hist.axvline(float(np.mean(lens)), color="red", linestyle="--",
                            linewidth=1.2, label=f"mean = {np.mean(lens):.2f}")
            ax_hist.axvline(float(np.median(lens)), color="orange", linestyle=":",
                            linewidth=1.2, label=f"median = {np.median(lens):.2f}")
            ax_hist.legend(fontsize=8)
        else:
            ax_hist.text(0.5, 0.5, "No multi-pixel branches", ha="center", va="center",
                         transform=ax_hist.transAxes)
    ax_hist.set_title("Branch-length distribution [px]")
    ax_hist.set_xlabel("length [px]")
    ax_hist.set_ylabel("count")

    # ---- Bottom-right: top components by length -----------------------
    ax_comp = fig.add_subplot(gs[1, 1])
    if not component_df.empty and "skeleton_length_px" in component_df.columns:
        top = component_df.nlargest(10, "skeleton_length_px")
        labels = [f"#{int(c)}" for c in top["component_id"]]
        values = top["skeleton_length_px"].to_numpy(dtype=float)
        positions = np.arange(len(labels))
        bars = ax_comp.bar(positions, values, color="#55A868", edgecolor="black", alpha=0.85)
        ax_comp.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax_comp.set_xticks(positions)
        ax_comp.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax_comp.text(0.5, 0.5, "No components", ha="center", va="center",
                     transform=ax_comp.transAxes)
    ax_comp.set_title("Top components by skeleton length [px]")
    ax_comp.set_ylabel("length [px]")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Step 4 metrics report saved: %s", path)


def save_metrics_text_report(
    global_metrics: Dict[str, object],
    path: Path,
    validation: Optional[Dict[str, float]] = None,
) -> None:
    """Save a plain-text human-readable summary of all global metrics."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MITOCHONDRIAL NETWORK ANALYSIS - STEP 4 METRICS SUMMARY")
    lines.append("=" * 72)
    for group_name, items in _METRIC_GROUPS:
        lines.append("")
        lines.append(f"[{group_name}]")
        for label, key in items:
            lines.append(f"  {label:<35s} : {_format_value(global_metrics.get(key))}")

    extra_keys = set(global_metrics) - {k for _, items in _METRIC_GROUPS for _, k in items}
    if extra_keys:
        lines.append("")
        lines.append("[Other]")
        for k in sorted(extra_keys):
            lines.append(f"  {k:<35s} : {_format_value(global_metrics[k])}")

    if validation:
        lines.append("")
        lines.append("[Validation vs reference skeleton]")
        for k, v in validation.items():
            lines.append(f"  {k:<35s} : {_format_value(v)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Step 4 metrics text report saved: %s", path)
