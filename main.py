"""Car.Ma._Cell - mitochondrial network analysis pipeline.

Paired Fusion / Fission (or any two-cell) comparison of mitochondrial
fluorescence micrographs. The pipeline reproduces the canonical Fiji
workflow (Subtract Background + tubeness + auto-threshold + Skeletonize
2D/3D + Analyze Skeleton) and is meant to be driven either from the
command line or from the bundled Tk GUI (``gui.py``).

Command-line usage::

    python main.py <cellA_image> <cellA_um> <cellB_image> <cellB_um>

Example (the synthetic pair shipped in ``examples/``)::

    python main.py examples/synthetic_fused.png 6 \
                   examples/synthetic_fragmented.png 6
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from mitochondrial_network_analysis.deconvolution import (  # noqa: E402
    generate_gaussian_psf,
    generate_theoretical_psf,
    run_deconvolution,
)
from mitochondrial_network_analysis.graph_analysis import (  # noqa: E402
    build_skeleton_graph,
    classify_skeleton_nodes,
    extract_branches,
)
from mitochondrial_network_analysis.io import load_image  # noqa: E402
from mitochondrial_network_analysis.preprocessing import (  # noqa: E402
    normalize_image,
)
from mitochondrial_network_analysis.segmentation import (  # noqa: E402
    clean_binary_mask,
)
from mitochondrial_network_analysis.skeletonization import (  # noqa: E402
    skeletonize_network,
)
from mitochondrial_network_analysis.metrics import (  # noqa: E402
    compute_branch_metrics,
    compute_component_metrics,
    compute_global_metrics,
)
from mitochondrial_network_analysis.analyze_skeleton import (  # noqa: E402
    build_analyze_skeleton_table,
)


# ============================================================================
# Paths and defaults
# ============================================================================
_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "main_output"

# Default inputs: the synthetic example pair in examples/ (60 px bar = 6 um).
DEFAULT_FUSION_IMAGE = _HERE / "examples" / "synthetic_fused.png"
DEFAULT_FUSION_SCALEBAR_UM = 6.0
DEFAULT_FISSION_IMAGE = _HERE / "examples" / "synthetic_fragmented.png"
DEFAULT_FISSION_SCALEBAR_UM = 6.0

# Target physical resolution (um per pixel) for the rescale step.
# ``None`` -> use the finer of the two source resolutions, so the coarser
# image is enlarged to the finer image's grid.
TARGET_UM_PER_PX: float | None = None

# Length of the reference scale bar overlaid on comparison figures.
DISPLAY_SCALEBAR_UM = 15.0

# Default labels for the two cells in a paired analysis.
DEFAULT_LABEL_A = "Fusion"
DEFAULT_LABEL_B = "Fission"

# Colour pairs for the comparison bar chart. ``color`` is the Okabe-Ito
# colour-blind safe palette (Wong, Nature Methods, 2011).
COLOR_SCHEMES: dict[str, tuple[str, str]] = {
    "color": ("#0072B2", "#D55E00"),
    "mono":  ("#808080", "#000000"),
}
DEFAULT_COLOR_SCHEME = "color"


# ============================================================================
# Step 2 - Richardson-Lucy deconvolution and display
# ============================================================================
BG_RADIUS = 100
SMOOTH_SIGMA = 0.0
USE_UNSHARP = False
UNSHARP_RADIUS = 2.0
UNSHARP_AMOUNT = 0.6
# Richardson-Lucy deconvolution in Step 2 (when False, Step 2 passes the
# normalised input through unchanged).
USE_DECONV = True
PSF_SIZE = 11
PSF_SIGMA = 1.3
DECONV_ITERATIONS = 40

# PSF source: "manual" (Gaussian PSF_SIZE/PSF_SIGMA) or "theoretical"
# (sigma = 0.21 * lambda / NA, MitoNet convention).
PSF_MODE = "manual"
PSF_NA = 1.4
PSF_WAVELENGTH_EM_NM = 520.0
PSF_PIXEL_SIZE_NM = 100.0

PIXEL_SIZE_UM: float | None = None

DISPLAY_LOW_PCT = 1.0
DISPLAY_HIGH_PCT = 99.5
DISPLAY_GAMMA = 1.3
DISPLAY_CLAHE_KERNEL = 64
DISPLAY_CLAHE_CLIP = 0.01


# ============================================================================
# Step 3 - segmentation and skeletonization
# ============================================================================
# When True, replicate Fiji "Default" (IsoData) threshold on 8-bit
# Normalize-stretched image. The tubeness path (default) matches the Fiji
# reference within ~2% on skeleton voxels for the bundled examples.
FIJI_MODE = False
FIJI_BG_RADIUS = 50
FIJI_NORMALIZE_SATURATION_PCT = 0.35
FIJI_THRESHOLD_METHOD = "isodata"

# Tubeness-based pipeline parameters.
THRESHOLD_METHOD = "li"
THRESHOLD_FACTOR = 4.0
MIN_OBJECT_SIZE = 0
MIN_HOLE_SIZE = 0
CLOSING_RADIUS = 2
FRANGI_SIGMAS = (1.0, 1.5)
TOPHAT_RADIUS = 3
TOPHAT_THR_FACTOR = 0.35
INTENSITY_THR_FACTOR = 1.0
NUCLEUS_FLOOR_FACTOR = 0.5

# Permissive cell-envelope mask used as safety net around the cell body.
CELL_MASK_OTSU_FACTOR = 0.10
CELL_MASK_DILATION_RADIUS = 6
CELL_MASK_SMOOTH_SIGMA = 8.0
CELL_MASK_INTERIOR_EROSION = 0

# Legacy / package-API knobs (not used by the active path).
SAUVOLA_WINDOW = 9
SAUVOLA_K = 0.30
MITO_THR_FACTOR = 0.90
MITO_KEEP_PCT_MIN = 30.0
MITO_KEEP_PCT_MAX = 65.0
VESSEL_THR_FACTOR = 0.30
VESSEL_KEEP_PCT_MIN = 25.0
VESSEL_KEEP_PCT_MAX = 60.0

MIN_COMPONENT_SKELETON_PX = 0

# Scale-bar / text-annotation removal safety bounds.
ANNOTATION_BORDER_FRAC = 0.18
ANNOTATION_INTENSITY_PCT = 99.0
ANNOTATION_PAD_PX = 6


# ============================================================================
# Logging
# ============================================================================
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


# ============================================================================
# Image helpers
# ============================================================================
def _to_red_rgb(intensity: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` intensity image to a black->red RGB image."""
    rgb = np.zeros((*intensity.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(intensity, 0.0, 1.0)
    return rgb


def _stretch(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Percentile-based contrast stretch to ``[0, 1]``."""
    lo = float(np.percentile(image, low_pct))
    hi = float(np.percentile(image, high_pct))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    out = (image - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _mask_scalebar(image: np.ndarray) -> np.ndarray:
    """Zero out bright thin annotations in the bottom strip."""
    out = image.copy()
    h, w = image.shape
    if h < 20 or w < 20:
        return out

    border_h = max(1, int(h * ANNOTATION_BORDER_FRAC))
    bottom = out[h - border_h:, :]
    if bottom.size == 0:
        return out

    try:
        thr = float(np.percentile(image, ANNOTATION_INTENSITY_PCT))
    except Exception:  # noqa: BLE001
        return out
    if not np.isfinite(thr) or thr <= 0:
        return out

    bright = bottom > thr
    if not bright.any():
        return out

    from skimage.morphology import label as _label

    from mitochondrial_network_analysis.segmentation import (
        remove_small_objects_compat,
    )
    bright = remove_small_objects_compat(bright, 8)
    if not bright.any():
        return out

    labels = _label(bright, connectivity=2)
    pad = ANNOTATION_PAD_PX
    row_offset = h - border_h
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        comp_h = int(ys.max() - ys.min() + 1)
        comp_w = int(xs.max() - xs.min() + 1)
        if comp_h * comp_w > 0.5 * border_h * w:
            continue
        r0 = max(0, row_offset + int(ys.min()) - pad)
        r1 = min(h, row_offset + int(ys.max()) + 1 + pad)
        c0 = max(0, int(xs.min()) - pad)
        c1 = min(w, int(xs.max()) + 1 + pad)
        out[r0:r1, c0:c1] = 0.0
    return out


def _to_grayscale_fluorescence(image: np.ndarray) -> np.ndarray:
    """Return a 2-D intensity image, picking the brightest channel of RGB(A)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float32)
        means = [float(arr[..., c].mean()) for c in range(arr.shape[-1])]
        best = int(np.argmax(means))
        print(
            f"[INFO]  Multi-channel input: using channel {best} "
            f"(means={[round(m, 4) for m in means]})."
        )
        return arr[..., best].astype(np.float32)
    raise ValueError(f"Unsupported image dimensionality: shape={arr.shape}")


def _check_input_quiet(path: Path, log) -> None:
    log(f"[CHECK] Input image path : {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    log(f"[CHECK] Input image found ({path.stat().st_size / 1024:.1f} KB).")


# ============================================================================
# Scale-bar detection and rescaling to a common physical resolution
# ============================================================================
def _detect_scale_bar_length_px(
    image: np.ndarray,
    *,
    bottom_frac: float = 0.20,
    right_frac: float = 0.60,
    bright_pct: float = 99.0,
) -> tuple[int, tuple[int, int, int, int] | None]:
    """Detect the scale-bar pixel length in the bottom-right corner.

    Returns ``(bar_length_px, bbox)`` or ``(0, None)`` if no plausible bar
    is found.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.max(axis=-1)
    h, w = img.shape
    if h < 20 or w < 20:
        return 0, None
    r0 = int(h * (1.0 - bottom_frac))
    c0 = int(w * (1.0 - right_frac))
    roi = img[r0:, c0:]
    if roi.size == 0:
        return 0, None
    try:
        thr = float(np.percentile(img, bright_pct))
    except Exception:  # noqa: BLE001
        return 0, None
    if not np.isfinite(thr) or thr <= 0:
        return 0, None
    binr = roi >= thr

    best_len = 0
    best_row = -1
    best_cs = -1
    best_ce = -1
    for i in range(binr.shape[0]):
        row = binr[i]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        diffs = np.diff(idx)
        splits = np.where(diffs > 1)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, idx.size - 1]
        for s, e in zip(starts, ends):
            runlen = int(idx[e] - idx[s] + 1)
            if runlen > best_len:
                best_len = runlen
                best_row = i
                best_cs = int(idx[s])
                best_ce = int(idx[e])
    if best_len <= 0 or best_row < 0:
        return 0, None
    bbox = (r0 + best_row, c0 + best_cs, r0 + best_row, c0 + best_ce)
    return best_len, bbox


def _rescale_to_common_resolution(
    image: np.ndarray,
    src_um_per_px: float,
    target_um_per_px: float,
) -> np.ndarray:
    """Resample ``image`` from ``src_um_per_px`` to ``target_um_per_px``."""
    if src_um_per_px <= 0 or target_um_per_px <= 0:
        raise ValueError(
            f"Invalid pixel size(s): src={src_um_per_px}, "
            f"target={target_um_per_px}"
        )
    factor = float(src_um_per_px) / float(target_um_per_px)
    if abs(factor - 1.0) < 1e-3:
        return np.asarray(image, dtype=np.float32)
    from skimage.transform import rescale as _rescale

    out = _rescale(
        image.astype(np.float32, copy=False),
        scale=factor,
        order=3,
        mode="reflect",
        anti_aliasing=(factor < 1.0),
        preserve_range=True,
        channel_axis=None,
    )
    return out.astype(np.float32)


# ============================================================================
# Per-step figure savers
# ============================================================================
def _save_step2(deconvolved: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(_to_red_rgb(deconvolved), interpolation="nearest")
    ax.set_title("Step 2 - Deconvolved")
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _save_step3(skeleton: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    canvas = np.zeros((*skeleton.shape, 3), dtype=np.float32)
    canvas[skeleton] = 1.0
    ax.imshow(canvas, interpolation="nearest")
    ax.set_title("Step 3 - Skeleton")
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ============================================================================
# Step 4 - graph analysis and annotated figure
# ============================================================================
def analyze_network(skeleton: np.ndarray, log=print) -> dict:
    """Build the skeleton graph and compute node / branch statistics."""
    graph = build_skeleton_graph(skeleton)
    classes = classify_skeleton_nodes(graph)
    junctions = classes["junctions"]
    endpoints = classes["endpoints"]
    branches, _ = extract_branches(graph)

    n_junctions = len(junctions)
    n_endpoints = len(endpoints)
    n_branches = len(branches)
    lengths = np.array([b.length_px for b in branches], dtype=np.float64)
    mean_len = float(lengths.mean()) if lengths.size else 0.0
    median_len = float(np.median(lengths)) if lengths.size else 0.0

    degrees = dict(graph.degree())
    junctions_df = pd.DataFrame(
        [{"row": r, "col": c, "degree": int(degrees[(r, c)])} for (r, c) in junctions]
    )
    if not junctions_df.empty:
        log(
            f"[CHECK] Junction degree: mean={junctions_df['degree'].mean():.2f}, "
            f"min={int(junctions_df['degree'].min())}, "
            f"max={int(junctions_df['degree'].max())}"
        )

    branches_df = pd.DataFrame(
        [
            {
                "branch_id": b.branch_id,
                "component_id": b.component_id,
                "start_row": b.start_node[0],
                "start_col": b.start_node[1],
                "end_row": b.end_node[0],
                "end_col": b.end_node[1],
                "start_type": b.start_node_type,
                "end_type": b.end_node_type,
                "n_pixels": b.n_pixels,
                "length_px": b.length_px,
            }
            for b in branches
        ]
    )

    log(
        f"[CHECK] Network      : junctions={n_junctions}, endpoints={n_endpoints}, "
        f"branches={n_branches}, mean length={mean_len:.2f} px (median={median_len:.2f})"
    )

    h, w = skeleton.shape
    fig, ax = plt.subplots(figsize=(8, 8 * h / w), dpi=150, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    cmap = plt.get_cmap("tab20")
    segments_per_color: dict = {}
    for b in branches:
        if len(b.pixels) < 2:
            continue
        pts = np.array([(c, r) for (r, c) in b.pixels])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        color = cmap(b.branch_id % 20)
        segments_per_color.setdefault(color, []).append(segs)
    for color, segs_list in segments_per_color.items():
        all_segs = np.concatenate(segs_list, axis=0)
        lc = LineCollection(all_segs, colors=[color], linewidths=1.2)
        ax.add_collection(lc)

    if junctions:
        jr = np.array([n[0] for n in junctions])
        jc = np.array([n[1] for n in junctions])
        ax.scatter(
            jc, jr, s=18, c="red", edgecolors="white", linewidths=0.4,
            zorder=3, label=f"Junctions ({n_junctions})",
        )
    if endpoints:
        er = np.array([n[0] for n in endpoints])
        ec = np.array([n[1] for n in endpoints])
        ax.scatter(
            ec, er, s=10, c="cyan", edgecolors="white", linewidths=0.3,
            zorder=3, label=f"Endpoints ({n_endpoints})",
        )

    title = (
        f"Step 4 - Network: {n_junctions} junctions, {n_endpoints} endpoints, "
        f"{n_branches} branches\n"
        f"mean branch length = {mean_len:.1f} px (median = {median_len:.1f} px)"
    )
    ax.set_title(title, color="white", fontsize=10)
    if junctions or endpoints:
        leg = ax.legend(
            loc="lower right", facecolor="black", edgecolor="white",
            labelcolor="white", fontsize=8,
        )
        leg.get_frame().set_alpha(0.7)
    fig.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()

    # ---- Pixel-registered overlay (same grid as the skeleton) --------------
    # The annotated ``rgb`` above is a rendered figure (with title/legend/
    # margins), so its pixel dimensions do NOT match the skeleton. For the GUI
    # we also build an overlay on the *exact* skeleton grid: colour-coded
    # branches with red junction / cyan endpoint markers. Because it shares the
    # skeleton's shape, it displays at the very same physical scale as the
    # Step 2 / Step 3 images.
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.float32)
    for b in branches:
        if not b.pixels:
            continue
        color = np.asarray(cmap(b.branch_id % 20)[:3], dtype=np.float32)
        rr = np.fromiter((p[0] for p in b.pixels), dtype=np.intp, count=len(b.pixels))
        cc = np.fromiter((p[1] for p in b.pixels), dtype=np.intp, count=len(b.pixels))
        overlay[rr, cc] = color

    def _stamp(nodes, color, radius: int = 1) -> None:
        h_, w_ = overlay.shape[:2]
        col = np.asarray(color, dtype=np.float32)
        for r, c in nodes:
            r0, r1 = max(0, r - radius), min(h_, r + radius + 1)
            c0, c1 = max(0, c - radius), min(w_, c + radius + 1)
            overlay[r0:r1, c0:c1] = col

    _stamp(endpoints, (0.0, 1.0, 1.0), radius=1)   # cyan tips
    _stamp(junctions, (1.0, 0.0, 0.0), radius=1)   # red branch points (on top)

    return {
        "n_junctions": n_junctions,
        "n_endpoints": n_endpoints,
        "n_branches": n_branches,
        "mean_branch_length_px": mean_len,
        "median_branch_length_px": median_len,
        "branches_df": branches_df,
        "junctions_df": junctions_df,
        "fig": fig,
        "rgb": rgb,
        "overlay_rgb": overlay,
    }


# ============================================================================
# Step 5 - clean mitochondrial graph structure (node-link diagram)
# ============================================================================
# Colours follow the Okabe-Ito colour-blind-safe palette. The representation
# reproduces the canonical MitoGraph-style node-link graph seen in the
# literature (e.g. Mostov et al. 2026, Fig 1B; Wang et al. MitoTNT 2023):
# thin edges tracing each mitochondrial segment, with filled-circle nodes at
# branch points (junctions) and free tips (endpoints).
GRAPH_JUNCTION_COLOR = "#D55E00"    # vermillion - branch points (deg >= 3)
GRAPH_ENDPOINT_COLOR = "#0072B2"    # blue       - free tips     (deg == 1)
GRAPH_LARGEST_EDGE_COLOR = "#009E73"  # bluish green - largest connected network


def build_graph_structure(skeleton, graph, branches, components_sorted) -> dict:
    """Assemble a lightweight, drawable description of the skeleton graph.

    Returns a payload dict with the branch polylines (as edges), the junction /
    endpoint / isolated node coordinates, and summary counts. Coordinates are
    in ``(row, col)`` pixel units; the largest connected component (index 0 of
    ``components_sorted``, which is size-sorted) is flagged so it can be
    highlighted.
    """
    degree = dict(graph.degree())

    def _coords(nodes) -> np.ndarray:
        return np.array(list(nodes), dtype=float).reshape(-1, 2)

    edges = []
    for b in branches:
        if len(b.pixels) >= 2:
            pts = np.asarray(b.pixels, dtype=float)  # (N, 2) = (row, col)
            edges.append((pts, bool(b.component_id == 0)))

    junctions = _coords(n for n, d in degree.items() if d >= 3)
    endpoints = _coords(n for n, d in degree.items() if d == 1)
    isolated = _coords(n for n, d in degree.items() if d == 0)

    has_highlight = len(components_sorted) > 1 and any(is_l for _, is_l in edges)
    return {
        "shape": tuple(skeleton.shape),
        "edges": edges,
        "junctions": junctions,
        "endpoints": endpoints,
        "isolated": isolated,
        "n_components": len(components_sorted),
        "n_junctions": int(len(junctions)),
        "n_endpoints": int(len(endpoints)),
        "largest_size": len(components_sorted[0]) if components_sorted else 0,
        "has_largest_highlight": bool(has_highlight),
    }


def draw_graph_structure(
    ax,
    payload: dict,
    *,
    px: float = 1.0,
    edge_color: str = "#8A8A8E",
    edge_highlight: str = GRAPH_LARGEST_EDGE_COLOR,
    junction_color: str = GRAPH_JUNCTION_COLOR,
    endpoint_color: str = GRAPH_ENDPOINT_COLOR,
    bg: str | None = "white",
    title: str | None = None,
    title_color: str = "black",
    show_legend: bool = True,
) -> tuple[float, float, float, float]:
    """Draw the node-link graph onto ``ax`` and return its ``(l, r, b, t)`` extent.

    ``px`` scales pixel coordinates into physical units so the graph can share
    the same scale as the other steps. Shared by the GUI (theme colours) and
    the on-disk figure savers (publication colours).
    """
    ax.clear()
    if bg is not None:
        ax.set_facecolor(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    h, w = payload["shape"]
    extent = (0.0, w * px, h * px, 0.0)

    # Only highlight the largest network when there is more than one component
    # (otherwise the whole graph would be recoloured with no legend entry).
    highlight = bool(payload.get("has_largest_highlight"))
    normal_segs, hi_segs = [], []
    for pts, is_largest in payload["edges"]:
        xy = np.column_stack([pts[:, 1] * px, pts[:, 0] * px])  # (x=col, y=row)
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        (hi_segs if (is_largest and highlight) else normal_segs).append(segs)
    if normal_segs:
        ax.add_collection(
            LineCollection(np.concatenate(normal_segs), colors=edge_color,
                           linewidths=1.2, alpha=0.9)
        )
    if hi_segs:
        ax.add_collection(
            LineCollection(np.concatenate(hi_segs), colors=edge_highlight,
                           linewidths=1.7, alpha=0.95)
        )

    iso = payload.get("isolated")
    if iso is not None and len(iso):
        ax.scatter(iso[:, 1] * px, iso[:, 0] * px, s=8, c=edge_color,
                   zorder=2, linewidths=0)
    ep = payload["endpoints"]
    if len(ep):
        ax.scatter(ep[:, 1] * px, ep[:, 0] * px, s=14, c=endpoint_color,
                   zorder=3, edgecolors=bg or "none", linewidths=0.3)
    jn = payload["junctions"]
    if len(jn):
        ax.scatter(jn[:, 1] * px, jn[:, 0] * px, s=34, c=junction_color,
                   zorder=4, edgecolors=bg or "none", linewidths=0.4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if title:
        ax.set_title(title, color=title_color, fontsize=10)

    if show_legend and (len(jn) or len(ep)):
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=junction_color, markeredgecolor="none",
                   markersize=7, label=f"Junctions ({len(jn)})"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=endpoint_color, markeredgecolor="none",
                   markersize=6, label=f"Endpoints ({len(ep)})"),
        ]
        if payload.get("has_largest_highlight"):
            handles.append(
                Line2D([0], [0], color=edge_highlight, lw=2.2,
                       label="Largest network")
            )
        leg = ax.legend(
            handles=handles, loc="lower right", fontsize=7, framealpha=0.55,
            facecolor=bg or "white", edgecolor=edge_color, labelcolor=title_color,
        )
        leg.set_zorder(5)
    return extent


# ============================================================================
# Single-image pipeline
# ============================================================================
def _apply_exclusion_regions(
    gray: np.ndarray, exclude_regions: list | None
) -> np.ndarray:
    """Zero-out user-defined rectangular regions in a grayscale image.

    ``exclude_regions`` is a list of normalised ``(x0, y0, x1, y1)`` rectangles
    with coordinates in ``[0, 1]`` (origin = top-left). Pixels inside any of
    these rectangles are set to 0 so they are ignored by the downstream
    segmentation/skeletonisation. Returns a copy; the input is left untouched.
    """
    if not exclude_regions:
        return gray
    out = np.array(gray, dtype=np.float32, copy=True)
    h, w = out.shape[:2]
    for region in exclude_regions:
        try:
            x0, y0, x1, y1 = (float(v) for v in region)
        except (TypeError, ValueError):
            continue
        # Normalise ordering and clamp to [0, 1].
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        c0 = int(np.clip(round(x0 * w), 0, w))
        c1 = int(np.clip(round(x1 * w), 0, w))
        r0 = int(np.clip(round(y0 * h), 0, h))
        r1 = int(np.clip(round(y1 * h), 0, h))
        if c1 > c0 and r1 > r0:
            out[r0:r1, c0:c1] = 0.0
    return out


def process_image(
    input_path: Path | None,
    output_dir: Path | None = None,
    *,
    save: bool = True,
    verbose: bool = True,
    prepared_image: np.ndarray | None = None,
    pixel_size_um: float | None = None,
    label: str | None = None,
    exclude_regions: list | None = None,
) -> dict:
    """Run the full pipeline on a single image.

    ``exclude_regions`` is an optional list of *normalised* rectangles
    ``(x0, y0, x1, y1)`` with coordinates in ``[0, 1]`` relative to the image
    width/height (origin = top-left). Any pixel inside one of these rectangles
    is set to background (0) before analysis, so cell debris or neighbouring
    cells can be removed from the quantification.
    """
    log_prefix = f"[{label}] " if label else ""
    if verbose:
        def log(*a, **k):
            if a:
                a = (log_prefix + str(a[0]),) + a[1:]
            print(*a, **k)
    else:
        def log(*a, **k):
            return None

    input_path = Path(input_path).expanduser().resolve() if input_path else None
    if prepared_image is None:
        if input_path is None:
            raise ValueError(
                "process_image requires either 'input_path' or 'prepared_image'."
            )
        _check_input_quiet(input_path, log)

    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[CHECK] Output directory : {out_dir.resolve()}")

    # ---- Step 1 - load and normalize ---------------------------------------
    if prepared_image is not None:
        gray = np.asarray(prepared_image, dtype=np.float32)
        if gray.ndim != 2:
            raise ValueError(
                f"prepared_image must be 2-D, got shape {gray.shape}."
            )
    else:
        raw = load_image(input_path)
        gray = _to_grayscale_fluorescence(raw)
    gray = _apply_exclusion_regions(gray, exclude_regions)
    original = normalize_image(gray)
    original = _mask_scalebar(original)
    log(
        f"[CHECK] Loaded image : shape={original.shape}, "
        f"dtype={original.dtype}, "
        f"min={float(original.min()):.4f}, "
        f"max={float(original.max()):.4f}"
    )
    if original.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {original.shape}."
        )
    if not np.isfinite(original).all():
        raise ValueError("Input image contains NaN/Inf values after normalization.")

    # ---- Step 2 - Richardson-Lucy deconvolution + display ------------------
    if USE_DECONV:
        if PSF_MODE.lower() == "theoretical":
            psf = generate_theoretical_psf(
                numerical_aperture=PSF_NA,
                wavelength_em_nm=PSF_WAVELENGTH_EM_NM,
                pixel_size_nm=PSF_PIXEL_SIZE_NM,
            )
        else:
            psf = generate_gaussian_psf(PSF_SIZE, PSF_SIGMA)
        deconvolved = run_deconvolution(
            original, psf, iterations=DECONV_ITERATIONS, clip=False
        )
        deconvolved = normalize_image(deconvolved)
    else:
        deconvolved = original.astype(np.float32, copy=True)
    deconvolved_disp = _stretch(deconvolved, DISPLAY_LOW_PCT, DISPLAY_HIGH_PCT)
    if DISPLAY_GAMMA != 1.0:
        deconvolved_disp = np.power(deconvolved_disp, DISPLAY_GAMMA).astype(np.float32)
    log(
        f"[CHECK] Deconvolved  : shape={deconvolved.shape}, "
        f"min={float(deconvolved.min()):.4f}, "
        f"max={float(deconvolved.max()):.4f}"
    )

    step2_rgb = _to_red_rgb(deconvolved_disp)
    step2_path: Path | None = None
    if save:
        step2_path = out_dir / "step2_deconvolved.png"
        _save_step2(deconvolved_disp, step2_path)
        log(f"[OK]    Saved Step 2 figure -> {step2_path.resolve()}")

    # ---- Step 3 - segmentation + skeletonization ---------------------------
    from skimage.restoration import rolling_ball as _rball
    from skimage.filters import (
        threshold_otsu as _otsu, threshold_li as _li,
        threshold_triangle as _tri, threshold_yen as _yen,
        threshold_mean as _mean, threshold_isodata as _isodata,
        threshold_minimum as _minimum,
        gaussian as _gauss2, sato as _sato, median as _median2d,
    )
    from skimage.morphology import (
        dilation as _dil, erosion as _ero, disk as _disk,
    )
    from skimage.util import img_as_ubyte as _img_as_ubyte

    intensity_8bit: np.ndarray | None = None

    if FIJI_MODE:
        deconvolved_seg = deconvolved.astype(np.float32)
        seg_u8 = _img_as_ubyte(np.clip(deconvolved_seg, 0.0, 1.0))
        if FIJI_BG_RADIUS > 0:
            bg = _rball(seg_u8, radius=FIJI_BG_RADIUS)
            bgsub_u8 = np.clip(
                seg_u8.astype(np.int16) - bg.astype(np.int16), 0, 255
            ).astype(np.uint8)
        else:
            bgsub_u8 = seg_u8.copy()
        sat = max(0.0, float(FIJI_NORMALIZE_SATURATION_PCT))
        lo_p = sat / 2.0
        hi_p = 100.0 - sat / 2.0
        lo = float(np.percentile(bgsub_u8, lo_p))
        hi = float(np.percentile(bgsub_u8, hi_p))
        if hi <= lo:
            norm_u8 = bgsub_u8.copy()
        else:
            norm_f = (bgsub_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
            norm_u8 = np.clip(norm_f, 0.0, 255.0).astype(np.uint8)
        intensity_8bit = norm_u8
        _methods_fiji = {
            "isodata": _isodata, "otsu": _otsu, "li": _li,
            "triangle": _tri, "yen": _yen, "mean": _mean,
        }
        thr_fn = _methods_fiji.get(FIJI_THRESHOLD_METHOD.lower(), _isodata)
        threshold = float(thr_fn(norm_u8))
        mask = norm_u8 >= threshold
        log(
            f"[INFO]  Fiji mode: 8-bit -> SubtractBG (r={FIJI_BG_RADIUS}) "
            f"-> Normalize (sat={FIJI_NORMALIZE_SATURATION_PCT}%) "
            f"-> Auto-Threshold '{FIJI_THRESHOLD_METHOD}' = {threshold:.1f}"
        )
    else:
        deconvolved_seg = deconvolved.astype(np.float32)
        smoothed_env = _gauss2(
            deconvolved_seg, sigma=CELL_MASK_SMOOTH_SIGMA, preserve_range=True
        ).astype(np.float32)
        cell_thr = float(_otsu(smoothed_env))
        cell_mask = smoothed_env > (cell_thr * CELL_MASK_OTSU_FACTOR)
        cell_frac = float(cell_mask.mean())
        if cell_frac < 0.01 or cell_frac > 0.95:
            cell_mask = smoothed_env > cell_thr
            log(
                f"[WARN]  Relaxed cell mask degenerate (fraction={cell_frac:.3f}), "
                f"falling back to plain Otsu (threshold={cell_thr:.4f})."
            )
        if CELL_MASK_DILATION_RADIUS > 0:
            cell_mask = _dil(cell_mask, footprint=_disk(CELL_MASK_DILATION_RADIUS))
        if CELL_MASK_INTERIOR_EROSION > 0:
            cell_mask = _ero(cell_mask, footprint=_disk(CELL_MASK_INTERIOR_EROSION))
        log(
            f"[CHECK] Cell mask    : fraction={float(cell_mask.mean()):.3f} "
            f"(otsu={cell_thr:.4f}, factor={CELL_MASK_OTSU_FACTOR})"
        )

        if BG_RADIUS > 0:
            bg = _rball(deconvolved_seg, radius=BG_RADIUS).astype(np.float32)
            bgsub = np.clip(deconvolved_seg - bg, 0.0, None).astype(np.float32)
        else:
            bgsub = deconvolved_seg.astype(np.float32)
        bgsub_med = _median2d(bgsub, footprint=_disk(1)).astype(np.float32)
        tubeness = _sato(
            bgsub_med, sigmas=FRANGI_SIGMAS, black_ridges=False, mode="reflect"
        ).astype(np.float32)
        t_max = float(tubeness.max())
        if t_max > 0:
            tubeness = tubeness / t_max
        sharpened_seg = tubeness

        _methods = {
            "otsu": _otsu, "li": _li, "triangle": _tri,
            "yen": _yen, "mean": _mean, "isodata": _isodata,
            "minimum": _minimum,
        }
        thr_fn = _methods.get(THRESHOLD_METHOD.lower(), _otsu)
        try:
            base_threshold = float(thr_fn(sharpened_seg))
        except RuntimeError as exc:
            # threshold_minimum needs a histogram with two maxima.
            log(f"[WARN]  '{THRESHOLD_METHOD}' threshold failed ({exc}); using Otsu.")
            base_threshold = float(_otsu(sharpened_seg))
        threshold = base_threshold * THRESHOLD_FACTOR
        cell_vals_dec = deconvolved_seg[cell_mask]
        # Guard against a degenerate (empty) cell envelope: threshold_otsu on an
        # empty array raises IndexError.
        nucleus_floor = (
            float(_otsu(cell_vals_dec)) * NUCLEUS_FLOOR_FACTOR
            if cell_vals_dec.size else 0.0
        )
        mask = (
            (sharpened_seg > threshold)
            & (deconvolved_seg > nucleus_floor)
            & cell_mask
        )
        log(
            f"[INFO]  Subtract BG (r={BG_RADIUS}) + Sato tubeness "
            f"(sigmas={FRANGI_SIGMAS}) + Auto-Threshold "
            f"'{THRESHOLD_METHOD}' x {THRESHOLD_FACTOR} = {threshold:.4f}; "
            f"nucleus floor (Otsu*{NUCLEUS_FLOOR_FACTOR}) = {nucleus_floor:.4f}"
        )

    mask = clean_binary_mask(
        mask,
        min_object_size=MIN_OBJECT_SIZE,
        min_hole_size=MIN_HOLE_SIZE,
        apply_opening=False,
        apply_closing=(CLOSING_RADIUS > 0),
        closing_radius=CLOSING_RADIUS,
    )
    n_fg = int(mask.sum())
    log(
        f"[CHECK] Mito mask    : foreground pixels = {n_fg} "
        f"(mode={'Fiji-faithful' if FIJI_MODE else 'tubeness'})"
    )
    if n_fg == 0:
        raise RuntimeError(
            "Binary mask is empty - thresholding removed all signal."
        )

    skeleton = skeletonize_network(mask)
    if MIN_COMPONENT_SKELETON_PX > 0 and skeleton.any():
        from skimage.measure import label as _label
        lbl = _label(skeleton, connectivity=2)
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= MIN_COMPONENT_SKELETON_PX
        keep[0] = False
        skeleton = keep[lbl]
    n_skel = int(skeleton.sum())
    log(f"[CHECK] Skeleton     : pixels = {n_skel}")
    if n_skel == 0:
        raise RuntimeError("Skeleton is empty - check segmentation parameters.")

    step3_rgb = np.zeros((*skeleton.shape, 3), dtype=np.float32)
    step3_rgb[skeleton] = 1.0
    step3_path: Path | None = None
    if save:
        step3_path = out_dir / "step3_skeleton.png"
        _save_step3(skeleton, step3_path)
        log(f"[OK]    Saved Step 3 figure -> {step3_path.resolve()}")

    # ---- Step 4 - graph metrics and figure ---------------------------------
    step4 = analyze_network(skeleton, log=log)

    from mitochondrial_network_analysis.graph_analysis import (
        build_skeleton_graph as _bsg,
        extract_branches as _eb,
    )
    _graph = _bsg(skeleton)
    _branches, _components_sorted = _eb(_graph)

    # ---- Step 5 - clean node-link graph structure --------------------------
    graph_structure = build_graph_structure(
        skeleton, _graph, _branches, _components_sorted
    )

    roi_area_px = int(skeleton.shape[0] * skeleton.shape[1])
    px_um = pixel_size_um if pixel_size_um is not None else PIXEL_SIZE_UM
    global_metrics = compute_global_metrics(
        skeleton,
        _graph,
        _branches,
        _components_sorted,
        pixel_size_um=px_um,
        roi_area_px=roi_area_px,
        mask=mask,
    )
    component_metrics_df = compute_component_metrics(
        _graph, _branches, _components_sorted, pixel_size_um=px_um
    )
    branch_metrics_df = compute_branch_metrics(
        _branches, _graph, _components_sorted, pixel_size_um=px_um
    )
    log(
        f"[CHECK] Global metrics: components={global_metrics['n_connected_components']}, "
        f"branching idx (1/px^2)={global_metrics['branching_index_per_px2']:.3e}, "
        f"connectivity={global_metrics['connectivity_index_length']:.3f}, "
        f"fragmentation={global_metrics['fragmentation_index_length']:.3f}, "
        f"complexity={global_metrics['network_complexity_index']:.3e}"
    )
    log(
        f"[CHECK] Morphology   : n_objects={global_metrics.get('n_objects_total', 0)}, "
        f"footprint={global_metrics.get('mito_footprint_pct', float('nan')):.2f}%, "
        f"fractal_dim={global_metrics.get('fractal_dimension', float('nan')):.3f}, "
        f"MNBN={global_metrics.get('mitochondrial_connectivity_index', float('nan')):.3f}, "
        f"n_networks={global_metrics.get('n_networks', 0)}, "
        f"n_individuals={global_metrics.get('n_individual_mitochondria', 0)}"
    )

    if intensity_8bit is None:
        from skimage.util import img_as_ubyte as _img_as_ubyte_fb
        intensity_for_table = _img_as_ubyte_fb(
            np.clip(deconvolved.astype(np.float32), 0.0, 1.0)
        )
    else:
        intensity_for_table = intensity_8bit
    analyze_skeleton_df = build_analyze_skeleton_table(
        _branches,
        _components_sorted,
        intensity_image=intensity_for_table,
        pixel_size_um=px_um,
    )

    step4_path: Path | None = None
    csv_branches: Path | None = None
    csv_junctions: Path | None = None
    if save:
        import json as _json
        step4_path = out_dir / "step4_network.png"
        step4["fig"].savefig(step4_path, facecolor="black", bbox_inches="tight")
        log(f"[OK]    Saved Step 4 figure -> {step4_path.resolve()}")
        csv_branches = out_dir / "step4_branches.csv"
        step4["branches_df"].to_csv(csv_branches, index=False)
        log(f"[OK]    Saved branch table  -> {csv_branches.resolve()}")
        csv_junctions = out_dir / "step4_junctions.csv"
        step4["junctions_df"].to_csv(csv_junctions, index=False)
        log(f"[OK]    Saved junction table-> {csv_junctions.resolve()}")
        gm_csv = out_dir / "step4_global_metrics.csv"
        pd.DataFrame([global_metrics]).to_csv(gm_csv, index=False)
        log(f"[OK]    Saved global metrics-> {gm_csv.resolve()}")
        gm_json = out_dir / "step4_global_metrics.json"
        with gm_json.open("w", encoding="utf-8") as _fh:
            _json.dump(global_metrics, _fh, indent=2, default=str)
        log(f"[OK]    Saved global metrics-> {gm_json.resolve()}")
        cm_csv = out_dir / "step4_component_metrics.csv"
        component_metrics_df.to_csv(cm_csv, index=False)
        log(f"[OK]    Saved component table-> {cm_csv.resolve()}")
        bm_csv = out_dir / "step4_branch_metrics.csv"
        branch_metrics_df.to_csv(bm_csv, index=False)
        log(f"[OK]    Saved branch metrics-> {bm_csv.resolve()}")
        as_csv = out_dir / "step4_analyze_skeleton.csv"
        analyze_skeleton_df.to_csv(as_csv, index=False)
        log(f"[OK]    Saved Analyze Skeleton table -> {as_csv.resolve()}")
    plt.close(step4["fig"])

    # ---- Step 5 figure (standalone, publication style) ---------------------
    step5_path: Path | None = None
    if save:
        figG, axG = plt.subplots(figsize=(7, 7), dpi=150)
        figG.patch.set_facecolor("white")
        draw_graph_structure(
            axG, graph_structure, px=1.0, bg="white",
            title="Step 5 - Graph structure", title_color="black",
        )
        step5_path = out_dir / "step5_graph.png"
        figG.tight_layout()
        figG.savefig(step5_path, facecolor="white", bbox_inches="tight")
        plt.close(figG)
        log(f"[OK]    Saved Step 5 figure -> {step5_path.resolve()}")

    return {
        "step2_rgb": step2_rgb,
        "step3_rgb": step3_rgb,
        "step4_rgb": step4["rgb"],
        "step4_overlay_rgb": step4["overlay_rgb"],
        "graph_structure": graph_structure,
        "step2_path": step2_path,
        "step3_path": step3_path,
        "step4_path": step4_path,
        "step5_path": step5_path,
        "threshold": threshold,
        "n_foreground": n_fg,
        "n_skeleton": n_skel,
        "n_junctions": step4["n_junctions"],
        "n_endpoints": step4["n_endpoints"],
        "n_branches": step4["n_branches"],
        "mean_branch_length_px": step4["mean_branch_length_px"],
        "branches_df": step4["branches_df"],
        "junctions_df": step4["junctions_df"],
        "global_metrics": global_metrics,
        "component_metrics_df": component_metrics_df,
        "branch_metrics_df": branch_metrics_df,
        "analyze_skeleton_df": analyze_skeleton_df,
        "label": label,
        "pixel_size_um": px_um,
        "skeleton": skeleton,
        "mito_mask": mask,
        "deconvolved_disp": deconvolved_disp,
    }


# ============================================================================
# Paired analysis (cell A vs cell B)
# ============================================================================
def _draw_scalebar(
    ax,
    image_shape: tuple[int, int],
    pixel_size_um: float,
    bar_um: float,
    *,
    color: str = "white",
    pad_frac: float = 0.04,
) -> None:
    """Overlay a horizontal scale bar of ``bar_um`` micrometres."""
    if pixel_size_um is None or pixel_size_um <= 0 or bar_um <= 0:
        return
    h, w = image_shape[:2]
    bar_px = bar_um / float(pixel_size_um)
    if bar_px <= 0 or bar_px > 0.9 * w:
        return
    pad_x = pad_frac * w
    pad_y = pad_frac * h
    y = h - pad_y
    x1 = w - pad_x
    x0 = x1 - bar_px
    ax.plot([x0, x1], [y, y], color=color, linewidth=3.0, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2.0,
        y - 0.012 * h,
        f"{bar_um:g} \u00b5m",
        color=color,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def _save_pair_panel(
    fusion: np.ndarray,
    fission: np.ndarray,
    title_f: str,
    title_s: str,
    out_path: Path,
    *,
    pixel_size_um: float | None,
    bar_um: float,
    suptitle: str = "",
) -> None:
    """Save a 1x2 side-by-side panel for the two cells."""
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), dpi=180, facecolor="black"
    )
    for ax, img, ttl in zip(axes, [fusion, fission], [title_f, title_s]):
        ax.set_facecolor("black")
        ax.imshow(img, interpolation="nearest")
        ax.set_title(ttl, color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_scalebar(
            ax, img.shape, pixel_size_um or 0.0, bar_um, color="white"
        )
    if suptitle:
        fig.suptitle(suptitle, color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def _save_graph_pair_panel(
    payload_f: dict,
    payload_s: dict,
    title_f: str,
    title_s: str,
    out_path: Path,
    *,
    pixel_size_um: float | None,
    bar_um: float,
    suptitle: str = "",
) -> None:
    """Save a 1x2 side-by-side Step-5 graph panel (publication white style)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=180)
    fig.patch.set_facecolor("white")
    px = pixel_size_um or 1.0
    extents = []
    for ax, payload, ttl in zip(axes, [payload_f, payload_s], [title_f, title_s]):
        ext = draw_graph_structure(
            ax, payload, px=px, bg="white", title=ttl, title_color="black",
        )
        extents.append(ext)
    # Put both cells on a common physical scale.
    max_w = max(e[1] for e in extents)
    max_h = max(e[2] for e in extents)
    for ax in axes:
        ax.set_xlim(0.0, max_w)
        ax.set_ylim(max_h, 0.0)
        if pixel_size_um and bar_um > 0:
            _draw_scalebar(ax, (max_h, max_w), 1.0, bar_um, color="black")
    if suptitle:
        fig.suptitle(suptitle, color="black", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# Metrics promoted to the headline comparison bar chart.
_COMPARISON_METRICS = [
    ("mito_footprint_pct", "Mito footprint (%)"),
    ("n_networks", "Networks"),
    ("n_individual_mitochondria", "Individuals"),
    ("mitochondrial_connectivity_index", "MNBN"),
    ("fractal_dimension", "Fractal dim."),
    ("n_connected_components", "Skeleton CCs"),
    ("connectivity_index_length", "Connectivity"),
    ("fragmentation_index_length", "Fragmentation"),
    ("network_complexity_index", "Complexity"),
    ("mean_branch_length_um", "Mean branch length (\u00b5m)"),
]

# Visual style for the comparison bar chart.
BAR_WIDTH = 0.45        # narrower rectangles (default matplotlib is 0.8)
BAR_TITLE_FONTSIZE = 13
BAR_TICK_FONTSIZE = 12
BAR_VALUE_FONTSIZE = 11
BAR_SUPTITLE_FONTSIZE = 15


def _finite_metric_items(
    fusion_metrics: dict, fission_metrics: dict
) -> list[tuple[str, float, float]]:
    """Return ``[(label, fusion_value, fission_value), ...]`` for the headline
    metrics where both cells have a finite value."""
    items: list[tuple[str, float, float]] = []
    for key, lbl in _COMPARISON_METRICS:
        vf = fusion_metrics.get(key)
        vs = fission_metrics.get(key)
        try:
            vf_f = float(vf)
            vs_f = float(vs)
        except (TypeError, ValueError):
            continue
        if np.isfinite(vf_f) and np.isfinite(vs_f):
            items.append((lbl, vf_f, vs_f))
    return items


def _fmt_bar_value(v: float) -> str:
    """Compact, readable value label (ints stay integers)."""
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v):,}"
    return f"{v:.3g}"


def render_metric_bars(
    fig,
    fusion_metrics: dict,
    fission_metrics: dict,
    *,
    label_a: str = DEFAULT_LABEL_A,
    label_b: str = DEFAULT_LABEL_B,
    colors: tuple[str, str] | None = None,
    text_color: str = "#1D1D1F",
    subtext_color: str = "#6E6E73",
    grid_color: str = "#D2D2D7",
    face: str | None = "white",
    suptitle: str | None = None,
) -> None:
    """Draw a clean, flat grid of headline-metric bars onto ``fig``.

    Minimalist / Apple-flavoured styling: no bar outlines or chart junk, only
    faint horizontal guides, soft headroom for the value labels, and colours
    that carry the emphasis. Shared by the GUI (theme colours, drawn live) and
    the PNG export (publication white). Clears ``fig`` before drawing.
    """
    from matplotlib.ticker import MaxNLocator

    if colors is None:
        colors = COLOR_SCHEMES[DEFAULT_COLOR_SCHEME]
    color_a, color_b = colors
    items = _finite_metric_items(fusion_metrics, fission_metrics)

    fig.clear()
    if face is not None:
        fig.patch.set_facecolor(face)

    if not items:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No metrics to display", ha="center", va="center",
                color=subtext_color, fontsize=11)
        return

    n = len(items)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    x = np.array([0.0, 1.0])

    for i, (lbl, vf, vs) in enumerate(items):
        ax = axes[i]
        if face is not None:
            ax.set_facecolor(face)
        bars = ax.bar(x, [vf, vs], width=0.60, color=[color_a, color_b],
                      edgecolor="none", zorder=3)

        # Declutter: drop the box, keep a faint baseline + horizontal guides.
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.8, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="x", length=0, labelsize=9, colors=text_color, pad=4)
        ax.tick_params(axis="y", length=0, labelsize=8, colors=subtext_color)
        ax.set_xticks(x)
        ax.set_xticklabels([label_a, label_b])
        ax.set_xlim(-0.8, 1.8)
        ax.set_title(lbl, fontsize=11, fontweight="bold", color=text_color, pad=8)

        # Headroom so the value labels never collide with the title.
        vmax = max(vf, vs, 0.0)
        vmin = min(vf, vs, 0.0)
        span = (vmax - vmin) or (abs(vmax) or 1.0)
        ax.set_ylim(vmin, vmax + span * 0.22)
        for b, v, c in zip(bars, [vf, vs], [color_a, color_b]):
            ax.annotate(
                _fmt_bar_value(v),
                (b.get_x() + b.get_width() / 2.0, b.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=9, color=c,
            )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", color=text_color)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.96) if suptitle else None)
    except Exception:  # noqa: BLE001
        pass


def _save_metric_bar_chart(
    fusion_metrics: dict,
    fission_metrics: dict,
    out_path: Path,
    *,
    label_a: str = DEFAULT_LABEL_A,
    label_b: str = DEFAULT_LABEL_B,
    colors: tuple[str, str] | None = None,
) -> None:
    """Save the headline-metric bar chart (publication white style) to a PNG."""
    n = max(1, len(_finite_metric_items(fusion_metrics, fission_metrics)))
    nrows = (n + 1) // 2
    fig = plt.figure(figsize=(9, 2.5 * nrows + 0.4), dpi=160)
    render_metric_bars(
        fig, fusion_metrics, fission_metrics,
        label_a=label_a, label_b=label_b, colors=colors,
        text_color="#1D1D1F", subtext_color="#6E6E73", grid_color="#D2D2D7",
        face="white", suptitle=f"{label_a} vs {label_b} \u2014 key metrics",
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sanitize_label(label: str) -> str:
    """Return a filesystem- and column-safe version of ``label``."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(label).strip().lower()).strip("_")
    return cleaned or "cell"


def _build_comparison_table(
    fusion_metrics: dict,
    fission_metrics: dict,
    *,
    label_a: str = DEFAULT_LABEL_A,
    label_b: str = DEFAULT_LABEL_B,
) -> pd.DataFrame:
    """One row per metric; columns: ``metric``, ``<a>``, ``<b>``, delta, ratio."""
    col_a = _sanitize_label(label_a)
    col_b = _sanitize_label(label_b)
    delta_col = f"delta_{col_a}_minus_{col_b}"
    ratio_col = f"ratio_{col_a}_over_{col_b}"
    keys = sorted(set(fusion_metrics) | set(fission_metrics))
    rows = []
    for k in keys:
        vf = fusion_metrics.get(k)
        vs = fission_metrics.get(k)
        try:
            vf_f = float(vf) if vf is not None else float("nan")
            vs_f = float(vs) if vs is not None else float("nan")
        except (TypeError, ValueError):
            rows.append(
                {
                    "metric": k,
                    col_a: vf,
                    col_b: vs,
                    delta_col: None,
                    ratio_col: None,
                }
            )
            continue
        delta = vf_f - vs_f
        ratio = (
            vf_f / vs_f
            if np.isfinite(vs_f) and vs_f != 0.0
            else float("nan")
        )
        rows.append(
            {
                "metric": k,
                col_a: vf_f,
                col_b: vs_f,
                delta_col: delta,
                ratio_col: ratio,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Export helpers (pictures + single Excel workbook)
# ============================================================================
def _excel_value(v):
    """Coerce a metric value into something Excel stores cleanly.

    Ints stay ints, floats stay floats (NaN/Inf -> blank), bools stay bools,
    everything else is passed through (usually a short string). ``None`` and
    non-finite numbers become blank cells.
    """
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(f):
        return None
    return f


def _unique_headers(headers: list[str]) -> list[str]:
    """Return headers with duplicates disambiguated (``X`` -> ``X (2)``)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            out.append(f"{h} ({seen[h]})")
        else:
            seen[h] = 1
            out.append(h)
    return out


def build_metrics_dataframe(
    columns: list[tuple[str, dict]],
    ordered_items: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Assemble a name/value(s) table from one metrics dict per figure.

    Parameters
    ----------
    columns
        ``[(header, metrics_dict), ...]`` - one entry per analysed figure. The
        ``header`` (e.g. the cell label) becomes the column title above its
        values.
    ordered_items
        Optional ``[(display_name, key), ...]`` giving the preferred row order
        and human-readable names for known metrics. Any metric keys present in
        the data but not listed here are appended afterwards using their raw
        key as the name, so nothing is silently dropped.

    Returns
    -------
    pandas.DataFrame
        First column ``"Metric"`` (the data name), then one column per figure.
    """
    headers = _unique_headers([h for h, _ in columns])
    dicts = [d or {} for _, d in columns]

    items: list[tuple[str, str]] = list(ordered_items or [])
    listed_keys = {k for _, k in items}
    # Append data-present keys not already covered, preserving first-seen order.
    for d in dicts:
        for k in d.keys():
            if k not in listed_keys:
                listed_keys.add(k)
                items.append((str(k), k))

    rows = []
    for display, key in items:
        if not any(key in d for d in dicts):
            continue
        row = {"Metric": display}
        for header, d in zip(headers, dicts):
            row[header] = _excel_value(d.get(key))
        rows.append(row)

    return pd.DataFrame(rows, columns=["Metric", *headers])


def export_metrics_excel(
    path: Path | str,
    columns: list[tuple[str, dict]],
    ordered_items: list[tuple[str, str]] | None = None,
    *,
    sheet_name: str = "Metrics",
) -> Path:
    """Write a single ``.xlsx`` workbook: name column + one value column/figure.

    The header row is bold and centred, the first row is frozen, and column
    widths are auto-sized for readability.
    """
    path = Path(path)
    df = build_metrics_dataframe(columns, ordered_items)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        from openpyxl.styles import Alignment, Font

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            width = max(
                (len(str(c.value)) if c.value is not None else 0)
                for c in col_cells
            )
            letter = col_cells[0].column_letter
            ws.column_dimensions[letter].width = min(max(width + 2, 12), 60)
    return path


def export_pictures(src_dir: Path | str, dest_dir: Path | str) -> list[Path]:
    """Copy every PNG under ``src_dir`` into ``dest_dir`` (structure preserved).

    Returns the list of files written. Subfolder layout (e.g. per-cell folders)
    is preserved so same-named figures from the two cells don't collide.
    """
    import shutil

    src = Path(src_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for png in sorted(src.rglob("*.png")):
        rel = png.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, target)
        copied.append(target)
    return copied


def process_pair(
    fusion_path: Path,
    fusion_scalebar_um: float,
    fission_path: Path,
    fission_scalebar_um: float,
    output_dir: Path | None = None,
    *,
    target_um_per_px: float | None = None,
    display_scalebar_um: float = DISPLAY_SCALEBAR_UM,
    verbose: bool = True,
    label_a: str = DEFAULT_LABEL_A,
    label_b: str = DEFAULT_LABEL_B,
    color_scheme: str = DEFAULT_COLOR_SCHEME,
    exclude_regions_a: list | None = None,
    exclude_regions_b: list | None = None,
) -> dict:
    """Run the pipeline on two images and write comparison artefacts.

    ``exclude_regions_a`` / ``exclude_regions_b`` are optional lists of
    normalised ``(x0, y0, x1, y1)`` rectangles (coordinates in ``[0, 1]``,
    origin = top-left) marking image areas to ignore for each cell (e.g. debris
    or neighbouring-cell fragments).
    """
    log = print if verbose else (lambda *a, **k: None)

    fusion_path = Path(fusion_path).expanduser().resolve()
    fission_path = Path(fission_path).expanduser().resolve()
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    label_a = str(label_a) if label_a else DEFAULT_LABEL_A
    label_b = str(label_b) if label_b else DEFAULT_LABEL_B
    folder_a = _sanitize_label(label_a)
    folder_b = _sanitize_label(label_b)
    if folder_a == folder_b:
        folder_b = folder_b + "_b"
    if color_scheme not in COLOR_SCHEMES:
        log(
            f"[PAIR]   Unknown color_scheme={color_scheme!r}; "
            f"falling back to {DEFAULT_COLOR_SCHEME!r}."
        )
        color_scheme = DEFAULT_COLOR_SCHEME
    colors = COLOR_SCHEMES[color_scheme]

    log("=" * 70)
    log(f"[PAIR]  Paired {label_a}/{label_b} analysis")
    log(f"[PAIR]   {label_a:<8s}: {fusion_path}  (bar={fusion_scalebar_um} um)")
    log(f"[PAIR]   {label_b:<8s}: {fission_path} (bar={fission_scalebar_um} um)")
    log(f"[PAIR]   Output  : {out_dir.resolve()}")
    log(f"[PAIR]   Colours : {color_scheme}  {colors}")

    # ---- 1) Load --------------------------------------------------------
    raw_f = _to_grayscale_fluorescence(load_image(fusion_path))
    raw_s = _to_grayscale_fluorescence(load_image(fission_path))

    # ---- 2) Auto-detect scale bars -> source um/pixel -------------------
    bar_f_px, _ = _detect_scale_bar_length_px(raw_f)
    bar_s_px, _ = _detect_scale_bar_length_px(raw_s)
    if bar_f_px <= 0:
        raise RuntimeError(
            f"Could not detect a scale bar in {fusion_path.name}. "
            f"Check that a bright horizontal bar is present in the "
            f"bottom-right corner of the image."
        )
    if bar_s_px <= 0:
        raise RuntimeError(
            f"Could not detect a scale bar in {fission_path.name}."
        )
    um_per_px_f = float(fusion_scalebar_um) / float(bar_f_px)
    um_per_px_s = float(fission_scalebar_um) / float(bar_s_px)
    log(
        f"[PAIR]   {label_a} bar = {bar_f_px} px -> "
        f"{um_per_px_f * 1000:.1f} nm/px ({um_per_px_f:.5f} um/px)"
    )
    log(
        f"[PAIR]   {label_b} bar = {bar_s_px} px -> "
        f"{um_per_px_s * 1000:.1f} nm/px ({um_per_px_s:.5f} um/px)"
    )

    # ---- 3) Common physical resolution and rescale ----------------------
    if target_um_per_px is None:
        target = min(um_per_px_f, um_per_px_s)
        log(
            f"[PAIR]   Target um/px not given; using min "
            f"= {target:.5f} um/px (the finer of the two)."
        )
    else:
        target = float(target_um_per_px)
        log(f"[PAIR]   Target um/px (user-set) = {target:.5f}")

    img_f_rs = _rescale_to_common_resolution(raw_f, um_per_px_f, target)
    img_s_rs = _rescale_to_common_resolution(raw_s, um_per_px_s, target)
    log(
        f"[PAIR]   Rescaled shapes: {label_a} {raw_f.shape} -> {img_f_rs.shape}, "
        f"{label_b} {raw_s.shape} -> {img_s_rs.shape}"
    )

    # ---- 4) Per-cell pipelines ------------------------------------------
    log("-" * 70)
    log(f"[PAIR]  Processing {label_a} ...")
    res_f = process_image(
        None,
        out_dir / folder_a,
        save=True,
        verbose=verbose,
        prepared_image=img_f_rs,
        pixel_size_um=target,
        label=label_a,
        exclude_regions=exclude_regions_a,
    )
    log("-" * 70)
    log(f"[PAIR]  Processing {label_b} ...")
    res_s = process_image(
        None,
        out_dir / folder_b,
        save=True,
        verbose=verbose,
        prepared_image=img_s_rs,
        pixel_size_um=target,
        label=label_b,
        exclude_regions=exclude_regions_b,
    )
    log("-" * 70)

    # ---- 5) Comparison artefacts ----------------------------------------
    _save_pair_panel(
        res_f["step2_rgb"],
        res_s["step2_rgb"],
        title_f=f"{label_a} \u2014 Step 2 (deconvolved)",
        title_s=f"{label_b} \u2014 Step 2 (deconvolved)",
        out_path=out_dir / "comparison_step2.png",
        pixel_size_um=target,
        bar_um=display_scalebar_um,
        suptitle=f"Common scale: {target * 1000:.1f} nm/px",
    )
    log(
        f"[OK]    Saved comparison Step 2 -> "
        f"{(out_dir / 'comparison_step2.png').resolve()}"
    )

    _save_pair_panel(
        res_f["step3_rgb"],
        res_s["step3_rgb"],
        title_f=f"{label_a} \u2014 skeleton ({res_f['n_skeleton']} px)",
        title_s=f"{label_b} \u2014 skeleton ({res_s['n_skeleton']} px)",
        out_path=out_dir / "comparison_step3.png",
        pixel_size_um=target,
        bar_um=display_scalebar_um,
    )
    log(
        f"[OK]    Saved comparison Step 3 -> "
        f"{(out_dir / 'comparison_step3.png').resolve()}"
    )

    # Use the pixel-registered overlay (skeleton grid) rather than the rendered
    # step4 figure buffer, so the physical scale bar is drawn at the correct
    # length (the figure buffer has different pixel dimensions than the skeleton).
    _save_pair_panel(
        res_f["step4_overlay_rgb"],
        res_s["step4_overlay_rgb"],
        title_f=(
            f"{label_a} \u2014 networks={res_f['global_metrics'].get('n_networks', '?')}, "
            f"individuals={res_f['global_metrics'].get('n_individual_mitochondria', '?')}"
        ),
        title_s=(
            f"{label_b} \u2014 networks={res_s['global_metrics'].get('n_networks', '?')}, "
            f"individuals={res_s['global_metrics'].get('n_individual_mitochondria', '?')}"
        ),
        out_path=out_dir / "comparison_step4.png",
        pixel_size_um=target,
        bar_um=display_scalebar_um,
    )
    log(
        f"[OK]    Saved comparison Step 4 -> "
        f"{(out_dir / 'comparison_step4.png').resolve()}"
    )

    _save_graph_pair_panel(
        res_f["graph_structure"],
        res_s["graph_structure"],
        title_f=(
            f"{label_a} — graph: {res_f['graph_structure']['n_junctions']} junctions, "
            f"{res_f['graph_structure']['n_endpoints']} endpoints"
        ),
        title_s=(
            f"{label_b} — graph: {res_s['graph_structure']['n_junctions']} junctions, "
            f"{res_s['graph_structure']['n_endpoints']} endpoints"
        ),
        out_path=out_dir / "comparison_step5.png",
        pixel_size_um=target,
        bar_um=display_scalebar_um,
    )
    log(
        f"[OK]    Saved comparison Step 5 -> "
        f"{(out_dir / 'comparison_step5.png').resolve()}"
    )

    cmp_df = _build_comparison_table(
        res_f["global_metrics"], res_s["global_metrics"],
        label_a=label_a, label_b=label_b,
    )
    cmp_csv = out_dir / "comparison_metrics.csv"
    cmp_df.to_csv(cmp_csv, index=False)
    log(f"[OK]    Saved comparison metrics CSV -> {cmp_csv.resolve()}")

    _save_metric_bar_chart(
        res_f["global_metrics"],
        res_s["global_metrics"],
        out_dir / "comparison_metrics_bars.png",
        label_a=label_a,
        label_b=label_b,
        colors=colors,
    )
    log(
        f"[OK]    Saved comparison bar chart   -> "
        f"{(out_dir / 'comparison_metrics_bars.png').resolve()}"
    )

    log("-" * 70)
    log("[PAIR]  Headline comparison:")
    for key, lbl in _COMPARISON_METRICS:
        vf = res_f["global_metrics"].get(key)
        vs = res_s["global_metrics"].get(key)
        try:
            vf_s = f"{float(vf):.4g}"
            vs_s = f"{float(vs):.4g}"
        except (TypeError, ValueError):
            vf_s, vs_s = str(vf), str(vs)
        log(
            f"[PAIR]    {lbl:<32s} "
            f"{label_a}={vf_s:>10s}   {label_b}={vs_s:>10s}"
        )

    return {
        "fusion": res_f,
        "fission": res_s,
        "target_um_per_px": target,
        "fusion_um_per_px": um_per_px_f,
        "fission_um_per_px": um_per_px_s,
        "fusion_bar_px": bar_f_px,
        "fission_bar_px": bar_s_px,
        "comparison_df": cmp_df,
        "output_dir": out_dir,
        "label_a": label_a,
        "label_b": label_b,
        "color_scheme": color_scheme,
        "colors": colors,
    }


# ============================================================================
# CLI
# ============================================================================
def _resolve_pair_args(argv: list[str]) -> tuple[Path, float, Path, float]:
    """Parse ``<cellA> <umA> <cellB> <umB>`` from ``argv``."""
    args = list(argv[1:])
    if len(args) == 0:
        return (
            DEFAULT_FUSION_IMAGE,
            DEFAULT_FUSION_SCALEBAR_UM,
            DEFAULT_FISSION_IMAGE,
            DEFAULT_FISSION_SCALEBAR_UM,
        )
    if len(args) == 1:
        return (
            Path(args[0]),
            DEFAULT_FUSION_SCALEBAR_UM,
            DEFAULT_FISSION_IMAGE,
            DEFAULT_FISSION_SCALEBAR_UM,
        )
    if len(args) == 4:
        return (
            Path(args[0]),
            float(args[1]),
            Path(args[2]),
            float(args[3]),
        )
    raise SystemExit(
        "Usage:\n"
        "  python main.py <cellA_image> <cellA_um> "
        "<cellB_image> <cellB_um>\n"
        "Example:\n"
        "  python main.py examples/synthetic_fused.png 6 "
        "examples/synthetic_fragmented.png 6"
    )


def main() -> int:
    _setup_logging()

    # No command-line arguments -> the user is not running a scripted batch
    # job, so open the graphical interface (where images are chosen via the
    # Browse buttons). This makes ``python main.py`` "just work" and avoids
    # the confusing "Image not found" error caused by the missing example
    # images that ship with the source tree.
    if len(sys.argv) == 1:
        try:
            from gui import main as gui_main  # noqa: E402

            gui_main()
            return 0
        except Exception as exc:  # noqa: BLE001
            print(
                "\n[ERROR] Could not open the graphical interface:",
                exc,
                file=sys.stderr,
            )
            traceback.print_exc()
            return 1

    try:
        f_path, f_um, s_path, s_um = _resolve_pair_args(sys.argv)
        process_pair(
            f_path,
            f_um,
            s_path,
            s_um,
            output_dir=OUTPUT_DIR,
            target_um_per_px=TARGET_UM_PER_PX,
            display_scalebar_um=DISPLAY_SCALEBAR_UM,
            verbose=True,
        )
        print("\n[DONE] Paired pipeline finished successfully.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("\n[ERROR] Pipeline failed:", exc, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
