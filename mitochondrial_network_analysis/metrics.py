"""Quantitative metrics: global, per-component and per-branch."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from skimage.morphology import convex_hull_image

from .graph_analysis import (
    Branch,
    Pixel,
    classify_skeleton_nodes,
    component_is_isolated,
)
from .object_metrics import (
    compute_morphology_metrics,
    compute_network_partition_metrics,
    stats_dict,
)

logger = logging.getLogger(__name__)


def _safe(value: Optional[float]) -> Optional[float]:
    """Return ``None`` for ``None`` or NaN, otherwise the value as a float."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def estimate_cell_area_px(mask: np.ndarray) -> int:
    """Estimate the cell footprint as the convex hull area of the binary mask.

    The convex hull of the mitochondrial mask provides a robust, scale-aware
    proxy of the cellular area, allowing dimensionless normalisation of network
    metrics independently of the image / ROI size. Falls back to the mask area
    if the convex hull cannot be computed (e.g. empty mask).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return 0
    try:
        hull = convex_hull_image(mask)
        return int(hull.sum())
    except Exception:  # noqa: BLE001
        return int(mask.sum())


def compute_global_metrics(
    skeleton: np.ndarray,
    graph,
    branches: Sequence[Branch],
    components_sorted: Sequence[Sequence[Pixel]],
    pixel_size_um: Optional[float],
    roi_area_px: int,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute global, network-wide metrics.

    When ``mask`` is provided, the convex hull of the mask is used as a proxy
    for the cell area and several dimensionless indices are added (branching
    index, connectivity, fragmentation, complexity).
    """
    classes = classify_skeleton_nodes(graph)
    n_endpoints = len(classes["endpoints"])
    n_junctions = len(classes["junctions"])
    n_branches = len(branches)

    total_length_px = float(sum(b.length_px for b in branches))
    branch_lengths = np.array(
        [b.length_px for b in branches if b.n_pixels > 1], dtype=float
    )

    degree: Dict[Pixel, int] = dict(graph.degree())
    isolated_components = sum(
        1 for comp in components_sorted if component_is_isolated(comp, degree)
    )
    isolated_component_ids = {
        cid for cid, comp in enumerate(components_sorted)
        if component_is_isolated(comp, degree)
    }
    n_isolated_filaments = sum(
        1 for b in branches if b.component_id in isolated_component_ids
    )
    n_non_isolated = n_branches - n_isolated_filaments

    total_length_um = total_length_px * pixel_size_um if pixel_size_um else None
    roi_area_um2 = roi_area_px * (pixel_size_um ** 2) if pixel_size_um else None
    density_px = total_length_px / roi_area_px if roi_area_px > 0 else float("nan")
    density_um = (
        total_length_um / roi_area_um2 if (pixel_size_um and roi_area_um2) else None
    )

    metrics: Dict[str, Optional[float]] = {
        "skeleton_pixels": int(skeleton.sum()),
        "total_skeleton_length_px": total_length_px,
        "total_skeleton_length_um": _safe(total_length_um),
        "n_connected_components": len(components_sorted),
        "n_isolated_components": int(isolated_components),
        "n_endpoints": int(n_endpoints),
        "n_junction_nodes": int(n_junctions),
        "n_branch_points": int(n_junctions),
        "n_branches": int(n_branches),
        "n_isolated_filaments": int(n_isolated_filaments),
        "n_non_isolated_filaments": int(n_non_isolated),
        "mean_branch_length_px": float(branch_lengths.mean()) if branch_lengths.size else 0.0,
        "median_branch_length_px": float(np.median(branch_lengths)) if branch_lengths.size else 0.0,
        "min_branch_length_px": float(branch_lengths.min()) if branch_lengths.size else 0.0,
        "max_branch_length_px": float(branch_lengths.max()) if branch_lengths.size else 0.0,
        "std_branch_length_px": float(branch_lengths.std(ddof=0)) if branch_lengths.size else 0.0,
        "roi_area_px": int(roi_area_px),
        "network_density_px_per_px2": float(density_px),
        "pixel_size_um": _safe(pixel_size_um),
        "roi_area_um2": _safe(roi_area_um2),
        "network_density_um_per_um2": _safe(density_um),
    }
    if pixel_size_um:
        for key in (
            "mean_branch_length",
            "median_branch_length",
            "min_branch_length",
            "max_branch_length",
            "std_branch_length",
        ):
            metrics[f"{key}_um"] = metrics[f"{key}_px"] * pixel_size_um

    # ------------------------------------------------------------------
    # Dimensionless / cell-size-independent indices
    # ------------------------------------------------------------------
    # Cell-area proxy: convex hull of the binary mitochondrial mask. This is
    # robust to image cropping and ROI changes. If no mask is given, fall back
    # to the ROI area (less accurate biologically but always available).
    cell_area_px: Optional[int]
    if mask is not None:
        cell_area_px = estimate_cell_area_px(mask)
    else:
        cell_area_px = None

    metrics["cell_area_px"] = cell_area_px
    metrics["cell_area_um2"] = (
        cell_area_px * (pixel_size_um ** 2) if (cell_area_px and pixel_size_um) else None
    )
    mito_area_px = int(mask.sum()) if mask is not None else None
    metrics["mitochondrial_area_px"] = mito_area_px
    metrics["mitochondrial_area_fraction"] = (
        (mito_area_px / cell_area_px) if (mito_area_px is not None and cell_area_px) else None
    )

    norm_area_px = cell_area_px if cell_area_px else roi_area_px
    norm_area_um2 = (
        norm_area_px * (pixel_size_um ** 2) if (norm_area_px and pixel_size_um) else None
    )

    # --- Branching index ---------------------------------------------------
    # Junctions normalised by cell area (per px^2 and per um^2). We also expose
    # a per-skeleton-length variant (junctions per unit network length), which
    # is a purely topological intensity of branching.
    metrics["branching_index_per_px2"] = (
        n_junctions / norm_area_px if norm_area_px else float("nan")
    )
    metrics["branching_index_per_um2"] = (
        n_junctions / norm_area_um2 if norm_area_um2 else None
    )
    metrics["branches_per_px2"] = (
        n_branches / norm_area_px if norm_area_px else float("nan")
    )
    metrics["branches_per_um2"] = (
        n_branches / norm_area_um2 if norm_area_um2 else None
    )
    metrics["junctions_per_unit_length_px"] = (
        n_junctions / total_length_px if total_length_px > 0 else float("nan")
    )
    metrics["endpoints_per_unit_length_px"] = (
        n_endpoints / total_length_px if total_length_px > 0 else float("nan")
    )

    # --- Mitochondrial connectivity index ---------------------------------
    # Length-based: fraction of total skeleton length belonging to networked
    # (non-isolated) components. Equals 1 if every filament is part of a
    # branched network, 0 if every component is an isolated filament/fragment.
    networked_length_px = float(
        sum(
            b.length_px
            for b in branches
            if b.component_id not in isolated_component_ids
        )
    )
    metrics["connectivity_index_length"] = (
        networked_length_px / total_length_px if total_length_px > 0 else float("nan")
    )
    metrics["connectivity_index_count"] = (
        n_non_isolated / n_branches if n_branches > 0 else float("nan")
    )

    # --- Fragmentation index ----------------------------------------------
    # 1 - connectivity (length-based) plus a count-based variant and a
    # density-based variant (number of components per unit cell area).
    metrics["fragmentation_index_length"] = (
        1.0 - metrics["connectivity_index_length"]
        if isinstance(metrics["connectivity_index_length"], float)
        and not math.isnan(metrics["connectivity_index_length"])
        else float("nan")
    )
    metrics["fragmentation_index_count"] = (
        isolated_components / len(components_sorted) if components_sorted else float("nan")
    )
    metrics["components_per_px2"] = (
        len(components_sorted) / norm_area_px if norm_area_px else float("nan")
    )
    metrics["components_per_um2"] = (
        len(components_sorted) / norm_area_um2 if norm_area_um2 else None
    )

    # --- Network complexity index -----------------------------------------
    # Combines branching intensity (junctions per unit length), normalised
    # skeleton length (skeleton length per cell area), and connectivity. All
    # three sub-terms are dimensionless after normalisation by the cell area
    # / network length, hence the index is itself dimensionless and comparable
    # across cells of different sizes.
    branching_intensity = metrics["junctions_per_unit_length_px"]
    normalised_length = (
        total_length_px / norm_area_px if norm_area_px else float("nan")
    )
    connectivity = metrics["connectivity_index_length"]

    def _finite(v: Optional[float]) -> float:
        if v is None:
            return 0.0
        if isinstance(v, float) and math.isnan(v):
            return 0.0
        return float(v)

    metrics["normalised_skeleton_length"] = _finite(normalised_length)
    metrics["network_complexity_index"] = (
        _finite(branching_intensity)
        * _finite(normalised_length)
        * _finite(connectivity)
    )

    # ------------------------------------------------------------------
    # Node-degree distribution
    # ------------------------------------------------------------------
    # Total number of skeleton-graph nodes broken down by their degree
    # (number of 8-connected neighbours), with the corresponding percentage
    # over the total. Degree 1 = endpoints, degree 2 = slab nodes,
    # degree >= 3 = junctions (branch points). Reported up to the maximum
    # degree actually observed in the skeleton.
    degree_values = list(degree.values())
    n_total_nodes = len(degree_values)
    metrics["n_total_nodes"] = int(n_total_nodes)
    if n_total_nodes > 0:
        max_degree = int(max(degree_values))
        from collections import Counter as _Counter
        deg_counts = _Counter(degree_values)
        for d in range(1, max_degree + 1):
            cnt = int(deg_counts.get(d, 0))
            metrics[f"n_nodes_degree_{d}"] = cnt
            metrics[f"pct_nodes_degree_{d}"] = (
                100.0 * cnt / n_total_nodes
            )
        # Aggregated counts and percentages for "junctions with K branches"
        # (degree >= 3), useful for biological interpretation.
        metrics["n_nodes_degree_ge3"] = int(
            sum(1 for v in degree_values if v >= 3)
        )
        metrics["pct_nodes_degree_ge3"] = (
            100.0 * metrics["n_nodes_degree_ge3"] / n_total_nodes
        )

    # ------------------------------------------------------------------
    # Extended distribution statistics on branch lengths (skewness,
    # kurtosis, CV%, IQR, p25, p75 - in px and, if calibrated, um).
    # Adapted from MitoNet._stats.
    # ------------------------------------------------------------------
    branch_lengths_list = [float(b.length_px) for b in branches if b.n_pixels > 1]
    metrics.update(stats_dict(branch_lengths_list, "branch_length_px"))
    if pixel_size_um:
        metrics.update(
            stats_dict(
                [v * pixel_size_um for v in branch_lengths_list],
                "branch_length_um",
            )
        )

    # ------------------------------------------------------------------
    # MNBN / network-vs-individual partition (Valente et al. 2017).
    # ------------------------------------------------------------------
    partition = compute_network_partition_metrics(
        graph, branches, components_sorted, pixel_size_um=pixel_size_um
    )
    # Avoid clobbering branch_length_* keys already inserted above.
    for k, v in partition.items():
        if k.startswith("branch_length_"):
            continue
        metrics[k] = v

    # Per-mitochondrion morphology metrics (regionprops on the binary mask)
    # + box-counting fractal dimension. Adapted from
    # MitoNet.compute_morphology_metrics / _fractal_dimension.
    if mask is not None:
        morph = compute_morphology_metrics(mask, pixel_size_um=pixel_size_um)
        for k, v in morph.items():
            metrics[k] = v

        # Derived indices that depend on both morphology and skeleton.
        n_obj = int(morph.get("n_objects_total", 0) or 0)
        area_um2_total = morph.get("total_mito_area_um2", float("nan"))
        n_bp = int(n_junctions)
        n_ind = int(partition.get("n_individual_mitochondria", 0) or 0)
        metrics["branching_index_per_mito"] = (
            (n_bp / n_obj) if n_obj > 0 else float("nan")
        )
        metrics["branching_index_per_area_um2"] = (
            (n_bp / area_um2_total)
            if (isinstance(area_um2_total, (int, float))
                and area_um2_total
                and not math.isnan(area_um2_total))
            else float("nan")
        )
        metrics["fragmentation_index_objects"] = (
            (n_ind / n_obj) if n_obj > 0 else float("nan")
        )

    return metrics


def compute_component_metrics(
    graph,
    branches: Sequence[Branch],
    components_sorted: Sequence[Sequence[Pixel]],
    pixel_size_um: Optional[float],
) -> pd.DataFrame:
    """Compute per-connected-component metrics."""
    degree: Dict[Pixel, int] = dict(graph.degree())
    branches_per_component: Dict[int, List[Branch]] = {}
    for b in branches:
        branches_per_component.setdefault(b.component_id, []).append(b)

    rows = []
    for cid, comp in enumerate(components_sorted):
        nodes = list(comp)
        coords = np.array(nodes)
        rows_idx, cols_idx = coords[:, 0], coords[:, 1]
        comp_branches = branches_per_component.get(cid, [])
        skeleton_length_px = float(sum(b.length_px for b in comp_branches))
        rows.append(
            {
                "component_id": cid,
                "n_pixels": len(nodes),
                "skeleton_length_px": skeleton_length_px,
                "skeleton_length_um": (
                    skeleton_length_px * pixel_size_um if pixel_size_um else None
                ),
                "n_endpoints": sum(1 for n in nodes if degree[n] == 1),
                "n_junctions": sum(1 for n in nodes if degree[n] >= 3),
                "n_branches": len(comp_branches),
                "is_isolated_filament": bool(component_is_isolated(nodes, degree)),
                "bbox_min_row": int(rows_idx.min()),
                "bbox_min_col": int(cols_idx.min()),
                "bbox_max_row": int(rows_idx.max()),
                "bbox_max_col": int(cols_idx.max()),
                "centroid_row": float(rows_idx.mean()),
                "centroid_col": float(cols_idx.mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_branch_metrics(
    branches: Sequence[Branch],
    graph,
    components_sorted: Sequence[Sequence[Pixel]],
    pixel_size_um: Optional[float],
) -> pd.DataFrame:
    """Compute per-branch metrics."""
    degree: Dict[Pixel, int] = dict(graph.degree())
    isolated_flag = {
        cid: component_is_isolated(list(comp), degree)
        for cid, comp in enumerate(components_sorted)
    }

    rows = []
    for b in branches:
        rows.append(
            {
                "branch_id": b.branch_id,
                "component_id": b.component_id,
                "start_node_type": b.start_node_type,
                "end_node_type": b.end_node_type,
                "start_row": b.start_node[0],
                "start_col": b.start_node[1],
                "end_row": b.end_node[0],
                "end_col": b.end_node[1],
                "length_px": b.length_px,
                "length_um": b.length_px * pixel_size_um if pixel_size_um else None,
                "n_pixels": b.n_pixels,
                "is_in_isolated_filament": bool(isolated_flag.get(b.component_id, False)),
            }
        )
    return pd.DataFrame(rows)


def export_metrics(
    output_dir: Path,
    global_metrics: Dict[str, Optional[float]],
    component_df: pd.DataFrame,
    branch_df: pd.DataFrame,
) -> None:
    """Save global / component / branch metrics to CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([global_metrics]).to_csv(output_dir / "global_metrics.csv", index=False)
    with (output_dir / "global_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(global_metrics, fh, indent=2, default=str)

    component_df.to_csv(output_dir / "component_metrics.csv", index=False)
    branch_df.to_csv(output_dir / "branch_metrics.csv", index=False)
    logger.info("Metrics written to %s", output_dir)
