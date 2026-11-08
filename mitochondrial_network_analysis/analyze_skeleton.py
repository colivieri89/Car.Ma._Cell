"""Replicates the per-branch output of Fiji's *Analyze Skeleton (2D/3D)*
plugin (Arganda-Carreras et al. 2010), so users that follow the
ImageJ User Guide procedure can compare results directly.

Each row corresponds to one branch in the skeleton graph and contains:

* ``Skeleton ID``           - identifies the parent connected component
                              (1-based, matches Fiji's convention).
* ``Branch length``         - geodesic length of the branch (px and, when
                              calibrated, um).
* ``V1 x / V1 y / V1 z``    - coordinates of the first endpoint (z = 0 in
                              2-D images).
* ``V2 x / V2 y / V2 z``    - coordinates of the second endpoint.
* ``Euclidean distance``    - straight-line distance between V1 and V2.
* ``Running average length``- progressive mean of the branch lengths
                              within the same skeleton, in row order.
* ``Average intensity (inner 3rd)`` - mean intensity of the pixels lying
                              in the middle third of the branch path.
* ``Average intensity``     - mean intensity along the full branch path.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _branch_intensity(
    pixels: Sequence[tuple[int, int]],
    image: Optional[np.ndarray],
    inner_third_only: bool = False,
) -> float:
    """Return the mean intensity of *image* along *pixels*.

    When ``inner_third_only`` is True, only the central third of the path
    contributes (Fiji's "Average intensity (inner 3rd)" column).
    Returns NaN if *image* is None or the path is empty.
    """
    if image is None or not pixels:
        return float("nan")
    n = len(pixels)
    if inner_third_only:
        if n < 3:
            sub = list(pixels)
        else:
            a = n // 3
            b = n - a
            sub = list(pixels[a:b])
            if not sub:
                sub = [pixels[n // 2]]
    else:
        sub = list(pixels)
    rs = np.fromiter((p[0] for p in sub), dtype=np.intp, count=len(sub))
    cs = np.fromiter((p[1] for p in sub), dtype=np.intp, count=len(sub))
    vals = image[rs, cs]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals.astype(np.float64)))


def build_analyze_skeleton_table(
    branches: Sequence,
    components_sorted: Sequence,
    intensity_image: Optional[np.ndarray],
    pixel_size_um: Optional[float] = None,
) -> pd.DataFrame:
    """Build a Fiji "Analyze Skeleton 2D/3D"-equivalent DataFrame.

    Parameters
    ----------
    branches
        Iterable of :class:`mitochondrial_network_analysis.graph_analysis.Branch`
        objects produced by :func:`extract_branches`.
    components_sorted
        The component list returned alongside ``branches``; used only to
        establish the 1-based ``Skeleton ID`` and the row order, which is
        first by Skeleton ID and then by branch id within the skeleton
        (Fiji's natural ordering).
    intensity_image
        2-D array sampled to populate the two "Average intensity" columns.
        For a Fiji-faithful pipeline this is the 8-bit normalised image
        fed to the Default auto-threshold. Pass ``None`` to skip intensity
        sampling (NaN values).
    pixel_size_um
        Pixel size in micrometres. When provided, ``Branch length`` and
        ``Euclidean distance`` are reported in um in addition to pixels.
    """
    # Map component_id -> 1-based Skeleton ID following Fiji's convention.
    component_ids_in_order = list(range(len(components_sorted)))
    skeleton_id_of = {cid: i + 1 for i, cid in enumerate(component_ids_in_order)}

    rows = []
    for b in branches:
        v1y, v1x = b.start_node
        v2y, v2x = b.end_node
        eucl_px = float(np.hypot(v2y - v1y, v2x - v1x))
        avg_int = _branch_intensity(b.pixels, intensity_image, inner_third_only=False)
        avg_int_inner = _branch_intensity(b.pixels, intensity_image, inner_third_only=True)
        row = {
            "Skeleton ID": int(skeleton_id_of.get(b.component_id, 0)),
            "Branch": int(b.branch_id),
            "Branch length (px)": float(b.length_px),
            "V1 x": int(v1x),
            "V1 y": int(v1y),
            "V1 z": 0,
            "V2 x": int(v2x),
            "V2 y": int(v2y),
            "V2 z": 0,
            "Euclidean distance (px)": eucl_px,
            "Average intensity (inner 3rd)": avg_int_inner,
            "Average intensity": avg_int,
        }
        if pixel_size_um:
            row["Branch length"] = float(b.length_px) * pixel_size_um
            row["Euclidean distance"] = eucl_px * pixel_size_um
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df["Running average length"] = pd.Series(dtype=float)
        return df

    df = df.sort_values(["Skeleton ID", "Branch"], kind="stable").reset_index(drop=True)
    # Running average length: cumulative mean of branch lengths within the
    # same Skeleton ID, in row order (Fiji semantics).
    length_col = "Branch length" if "Branch length" in df.columns else "Branch length (px)"
    df["Running average length"] = (
        df.groupby("Skeleton ID")[length_col].expanding().mean().reset_index(level=0, drop=True)
    )
    # Reorder columns to mirror Fiji's plugin layout.
    preferred = [
        "Skeleton ID",
        "Branch",
        "Branch length",
        "Branch length (px)",
        "V1 x", "V1 y", "V1 z",
        "V2 x", "V2 y", "V2 z",
        "Euclidean distance",
        "Euclidean distance (px)",
        "Running average length",
        "Average intensity (inner 3rd)",
        "Average intensity",
    ]
    ordered = [c for c in preferred if c in df.columns]
    return df[ordered]
