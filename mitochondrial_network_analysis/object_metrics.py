"""Per-object morphology metrics, distribution statistics and fractal
dimension.

The implementations are adapted from Salvatore Aiello's MiNA-style ``MitoNet``
single-file pipeline (``MitoNet.py``: ``_stats``, ``_fractal_dimension`` and
``compute_morphology_metrics``), translated to fit this package's API while
preserving the original mathematical definitions:

* Areas, aspect ratios and circularities are computed on the labelled binary
  mask via :func:`skimage.measure.regionprops`.
* ``_stats`` returns a flat dict of descriptive statistics (n, mean, sd,
  median, min, max, CV%, skewness, kurtosis, p25, p75, IQR) for any numeric
  distribution.
* ``fractal_dimension`` is the vectorised box-counting estimator (slope of
  ``log N(s)`` vs ``log 1/s``).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


_STATS_KEYS = (
    "n", "mean", "sd", "median", "min", "max",
    "cv_pct", "skewness", "kurtosis", "p25", "p75", "iqr",
)


def stats_dict(values: Iterable[float], prefix: str) -> Dict[str, float]:
    """Return a flat dict of descriptive statistics for ``values``.

    Adapted from ``MitoNet._stats``. Keys are ``<prefix>_<stat>`` for every
    entry in :data:`_STATS_KEYS`. NaN-only / empty inputs yield NaNs.
    """
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return {f"{prefix}_{k}": nan for k in _STATS_KEYS}

    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    cv = (sd / mean * 100.0) if mean != 0 else float("nan")

    return {
        f"{prefix}_n": n,
        f"{prefix}_mean": mean,
        f"{prefix}_sd": sd,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_cv_pct": cv,
        f"{prefix}_skewness": float(stats.skew(arr)) if n > 2 else float("nan"),
        f"{prefix}_kurtosis": float(stats.kurtosis(arr)) if n > 2 else float("nan"),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def fractal_dimension(
    binary: np.ndarray,
    box_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
) -> float:
    """Box-counting fractal dimension of a binary mask (vectorised).

    Adapted from ``MitoNet._fractal_dimension``. Returns NaN if fewer than
    two valid box scales contain at least one occupied box (e.g. empty mask).
    """
    mask = binary.astype(bool)
    counts: List[int] = []
    for s in box_sizes:
        s = int(s)
        if s <= 0:
            continue
        h, w = mask.shape
        h2, w2 = (h // s) * s, (w // s) * s
        if h2 == 0 or w2 == 0:
            counts.append(0)
            continue
        block = mask[:h2, :w2].reshape(h2 // s, s, w2 // s, s)
        counts.append(int(block.any(axis=(1, 3)).sum()))

    valid = [(s, c) for s, c in zip(box_sizes, counts) if c > 0]
    if len(valid) < 2:
        return float("nan")

    xs = np.log([1.0 / s for s, _ in valid])
    ys = np.log([float(c) for _, c in valid])
    slope, *_ = stats.linregress(xs, ys)
    return float(slope)


def _regionprop_axes(p) -> tuple[float, float]:
    """Return (major, minor) axis length for a region, across skimage versions."""
    major = getattr(p, "axis_major_length", None)
    if major is None:
        major = p.major_axis_length
    minor = getattr(p, "axis_minor_length", None)
    if minor is None:
        minor = p.minor_axis_length
    return float(major), float(minor)


def compute_morphology_metrics(
    binary: np.ndarray,
    pixel_size_um: Optional[float] = None,
    fractal_box_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
) -> Dict[str, float]:
    """Per-object morphology metrics from the binary mask.

    Adapted from ``MitoNet.compute_morphology_metrics``. Computes per-region
    area, aspect ratio (major/minor axis) and circularity (``4 pi A / P^2``);
    aggregates each distribution via :func:`stats_dict`; reports the total
    mitochondrial footprint (px and %, plus um^2 if ``pixel_size_um`` given);
    and includes the box-counting fractal dimension of the mask.

    All ``*_um2`` keys are also reported in pixel units (``*_px``); when no
    pixel size is provided the um^2 entries are NaN but the pixel-based
    metrics are always populated.
    """
    mask = binary.astype(bool)
    labeled = label(mask)
    props = regionprops(labeled)

    px2_to_um2 = (pixel_size_um ** 2) if pixel_size_um else None

    areas_px = [float(p.area) for p in props]
    areas_um2 = (
        [a * px2_to_um2 for a in areas_px] if px2_to_um2 is not None else []
    )

    aspect_ratios: List[float] = []
    circularities: List[float] = []
    for p in props:
        major, minor = _regionprop_axes(p)
        aspect_ratios.append(major / minor if minor > 0 else float("nan"))
        perim = float(p.perimeter) if p.perimeter else 0.0
        circularities.append(
            4.0 * np.pi * float(p.area) / (perim ** 2) if perim > 0 else float("nan")
        )

    total_mito_px = float(sum(areas_px))
    image_px = float(mask.shape[0] * mask.shape[1])
    footprint_pct = (total_mito_px / image_px * 100.0) if image_px else float("nan")

    out: Dict[str, float] = {
        "n_objects_total": int(len(props)),
        "total_mito_area_px": total_mito_px,
        "image_area_px": image_px,
        "mito_footprint_pct": footprint_pct,
    }
    if px2_to_um2 is not None:
        out["total_mito_area_um2"] = total_mito_px * px2_to_um2
        out["image_area_um2"] = image_px * px2_to_um2
    else:
        out["total_mito_area_um2"] = float("nan")
        out["image_area_um2"] = float("nan")

    out.update(stats_dict(areas_px, "area_px"))
    if areas_um2:
        out.update(stats_dict(areas_um2, "area_um2"))
    else:
        out.update(stats_dict([], "area_um2"))
    out.update(stats_dict(aspect_ratios, "aspect_ratio"))
    out.update(stats_dict(circularities, "circularity"))

    out["fractal_dimension"] = fractal_dimension(mask, box_sizes=fractal_box_sizes)
    return out


def compute_network_partition_metrics(
    graph,
    branches,
    components_sorted,
    pixel_size_um: Optional[float] = None,
) -> Dict[str, float]:
    """Partition components into networks vs individual mitochondria and
    compute MNBN-style metrics.

    A *network* is a connected component containing at least one branchpoint
    (graph node of degree >= 3). An *individual mitochondrion* is a component
    with no branchpoints (linear filament or isolated pixel).

    Returns a dict with:

    * ``n_networks``, ``n_individual_mitochondria``
    * ``network_to_individual_ratio``
    * ``mitochondrial_connectivity_index`` (= MNBN, mean number of branches
      per network, Valente et al. 2017)
    * extended distribution stats for branch lengths (px and, if calibrated,
      um), per-network size and per-network branch count
    """
    degree = dict(graph.degree())
    branches_per_component: Dict[int, List] = {}
    for b in branches:
        branches_per_component.setdefault(b.component_id, []).append(b)

    n_networks = 0
    n_individuals = 0
    net_size_px: List[float] = []
    net_branches: List[int] = []
    for cid, comp in enumerate(components_sorted):
        has_branchpoint = any(degree.get(n, 0) >= 3 for n in comp)
        comp_branches = branches_per_component.get(cid, [])
        if has_branchpoint:
            n_networks += 1
            net_size_px.append(float(sum(b.length_px for b in comp_branches)))
            net_branches.append(len(comp_branches))
        else:
            n_individuals += 1

    out: Dict[str, float] = {
        "n_networks": int(n_networks),
        "n_individual_mitochondria": int(n_individuals),
        "network_to_individual_ratio": (
            (n_networks / n_individuals) if n_individuals > 0 else float("nan")
        ),
    }

    branch_lengths_px = [float(b.length_px) for b in branches if b.n_pixels > 1]
    out.update(stats_dict(branch_lengths_px, "branch_length_px"))
    if pixel_size_um:
        out.update(
            stats_dict(
                [v * pixel_size_um for v in branch_lengths_px],
                "branch_length_um",
            )
        )
        out.update(
            stats_dict([v * pixel_size_um for v in net_size_px], "network_size_um")
        )
    out.update(stats_dict(net_size_px, "network_size_px"))
    out.update(stats_dict(net_branches, "branches_per_network"))

    # MNBN = mean branches per network (NaN if no networks).
    out["mitochondrial_connectivity_index"] = out.get(
        "branches_per_network_mean", float("nan")
    )
    return out
