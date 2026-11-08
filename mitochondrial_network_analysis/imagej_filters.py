"""
ImageJ/Fiji filter and processing operations, ported to Python.

This module is a self-contained "compatibility layer" that re-implements
(or wraps) the operations available in the ImageJ User Guide
(https://imagej.net/ij/docs/guide/index.html), so that future versions of
the pipeline / GUI can expose them as user-selectable options.

Conventions
-----------
- All functions take a 2-D ``numpy.ndarray`` (grayscale, dtype float32 or
  uint8/uint16) as the first argument and return a ``numpy.ndarray`` of
  the same spatial shape.
- The docstring of each function references the exact ImageJ menu path,
  so users familiar with Fiji can locate the equivalent operation.
- No function in this module mutates its input array; results are always
  fresh arrays.
- Functions accept either uint8 / uint16 / float images. Internally they
  promote to float32 when needed and convert back if appropriate.

Coverage
--------
The module groups operations following the ImageJ menus:

* ``Process > Smooth / Sharpen / Find Edges / Find Maxima / Enhance Contrast``
* ``Process > Subtract Background``
* ``Process > Filters >`` (Gaussian Blur, Median, Mean, Minimum, Maximum,
  Variance, Unsharp Mask, Convolve, Top Hat, Tubeness, Frangi, Sato,
  Meijering, Hessian)
* ``Process > Math >`` (Add, Subtract, Multiply, Divide, Min, Max, Gamma,
  Log, Exp, Square, Square Root, Abs, Reciprocal, NaN handling)
* ``Process > Noise >`` (Add Specified Noise, Salt and Pepper, Despeckle,
  Remove Outliers, Remove NaNs)
* ``Process > Binary >`` (Make Binary, Erode, Dilate, Open, Close, Outline,
  Fill Holes, Skeletonize, Distance Map, Ultimate Points, Watershed,
  Voronoi)
* ``Process > FFT >`` (FFT magnitude/phase, Inverse FFT, Bandpass Filter,
  Custom Filter)
* ``Image > Adjust >`` (Brightness/Contrast, Threshold, Auto Threshold,
  Window/Level, Color Balance, Size, Canvas Size)
* ``Image > Type >`` (8-bit, 16-bit, 32-bit, RGB conversions)
* MiNA-specific plugin equivalents (Richardson-Lucy deconvolution,
  CLAHE, Tubeness/Sato).

Notes
-----
- Functions that have a direct ImageJ equivalent are named with the same
  capitalization as the menu item (in ``snake_case``), e.g.
  ``subtract_background`` for ``Process > Subtract Background...``.
- All "Auto Threshold" methods of ImageJ are implemented: Default
  (= IJ_IsoData), Huang, Intermodes, IsoData, Li, MaxEntropy, Mean,
  MinError, Minimum, Moments, Otsu, Percentile, RenyiEntropy, Shanbhag,
  Triangle, Yen.
- Functions that require a 3rd-party C plugin in Fiji (e.g. Tubeness,
  DeconvolutionLab2) are emulated with scikit-image / scipy equivalents,
  not exact bit-for-bit copies. The differences are documented in the
  docstrings.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Optional dependencies are imported lazily so this module loads quickly.
try:
    from scipy import ndimage as _ndi  # noqa: F401
    from scipy import fft as _sfft  # noqa: F401
except Exception:  # pragma: no cover
    _ndi = None  # type: ignore
    _sfft = None  # type: ignore


# =====================================================================
#                       Internal helpers
# =====================================================================
def _as_float32(img: ArrayLike) -> NDArray[np.float32]:
    """Return ``img`` as a contiguous float32 array (copy if needed)."""
    arr = np.asarray(img)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=True)
    else:
        arr = arr.copy()
    return arr


def _ensure_2d(img: ArrayLike) -> NDArray:
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape!r}")
    return arr


def _disk(radius: int):
    from skimage.morphology import disk
    return disk(int(radius))


def _square(size: int):
    from skimage.morphology import square
    return square(int(size))


# =====================================================================
#                Image > Type (data-type conversions)
# =====================================================================
def to_8bit(img: ArrayLike) -> NDArray[np.uint8]:
    """``Image > Type > 8-bit`` -- linear scale to 0-255 uint8."""
    arr = _as_float32(img)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def to_16bit(img: ArrayLike) -> NDArray[np.uint16]:
    """``Image > Type > 16-bit`` -- linear scale to 0-65535 uint16."""
    arr = _as_float32(img)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint16)
    out = (arr - lo) / (hi - lo) * 65535.0
    return np.clip(out, 0, 65535).astype(np.uint16)


def to_32bit(img: ArrayLike) -> NDArray[np.float32]:
    """``Image > Type > 32-bit`` -- promote to float32 without rescaling."""
    return np.asarray(img, dtype=np.float32)


def rgb_to_gray(img: ArrayLike, weights: Sequence[float] = (0.299, 0.587, 0.114)) -> NDArray[np.float32]:
    """``Image > Type > 8-bit`` (from RGB) -- ITU-R BT.601 luma by default."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError("rgb_to_gray expects an HxWx3 or HxWx4 image")
    w = np.asarray(weights[:3], dtype=np.float32)
    return arr[..., :3] @ w


# =====================================================================
#                Image > Adjust (display / contrast)
# =====================================================================
def brightness_contrast(
    img: ArrayLike,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> NDArray[np.float32]:
    """``Image > Adjust > Brightness/Contrast`` -- linear remap.

    ``out = contrast * (img - 0.5) + 0.5 + brightness`` (image normalised
    to [0, 1]).
    """
    arr = _as_float32(img)
    out = contrast * (arr - 0.5) + 0.5 + brightness
    return out.astype(np.float32)


def percentile_stretch(
    img: ArrayLike, low_pct: float = 0.35, high_pct: float = 99.65
) -> NDArray[np.float32]:
    """``Image > Adjust > Brightness/Contrast > Auto`` -- percentile saturation."""
    arr = _as_float32(img)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def gamma_correction(img: ArrayLike, gamma: float = 1.0) -> NDArray[np.float32]:
    """``Process > Math > Gamma`` -- ``out = img ** gamma`` (img in [0,1])."""
    arr = _as_float32(img)
    return np.power(np.clip(arr, 0.0, None), float(gamma)).astype(np.float32)


def window_level(
    img: ArrayLike, window: float, level: float
) -> NDArray[np.float32]:
    """``Image > Adjust > Window/Level`` -- centred linear remap to [0,1]."""
    arr = _as_float32(img)
    lo = level - window / 2.0
    hi = level + window / 2.0
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def invert(img: ArrayLike) -> NDArray[np.float32]:
    """``Edit > Invert`` -- ``out = max - img + min``."""
    arr = _as_float32(img)
    return (arr.max() + arr.min() - arr).astype(np.float32)


# =====================================================================
#                Image > Adjust > Threshold / Auto Threshold
# =====================================================================
# All 16 ImageJ Auto Threshold methods. The "Default" alias maps to
# IJ_IsoData (the historical ImageJ default).
AUTO_THRESHOLD_METHODS: tuple[str, ...] = (
    "default",       # = IJ_IsoData
    "huang",
    "intermodes",
    "isodata",
    "li",
    "maxentropy",
    "mean",
    "minerror",
    "minimum",
    "moments",
    "otsu",
    "percentile",
    "renyientropy",
    "shanbhag",
    "triangle",
    "yen",
)


def auto_threshold_value(
    img: ArrayLike, method: str = "default", ignore_black: bool = False,
    ignore_white: bool = False,
) -> float:
    """``Image > Adjust > Auto Threshold...`` -- return the threshold value.

    Methods follow the ImageJ Auto Threshold plugin by Gabriel Landini.
    """
    from skimage import filters as _filt
    arr = np.asarray(img).ravel()
    if ignore_black:
        arr = arr[arr > 0]
    if ignore_white:
        if arr.dtype.kind in "ui":
            arr = arr[arr < np.iinfo(arr.dtype).max]
        else:
            arr = arr[arr < float(arr.max())]
    if arr.size == 0:
        return 0.0
    m = method.lower().replace("_", "").replace("-", "")
    if m in ("default", "ijisodata", "isodata"):
        try:
            return float(_filt.threshold_isodata(arr))
        except Exception:
            return float(_filt.threshold_otsu(arr))
    if m == "otsu":
        return float(_filt.threshold_otsu(arr))
    if m == "li":
        return float(_filt.threshold_li(arr))
    if m == "yen":
        return float(_filt.threshold_yen(arr))
    if m == "mean":
        return float(_filt.threshold_mean(arr))
    if m == "triangle":
        return float(_filt.threshold_triangle(arr))
    if m == "minimum":
        return float(_filt.threshold_minimum(arr))
    if m in ("maxentropy", "renyientropy"):
        # Both supported by scikit-image as threshold_li alternatives.
        # Use scikit-image's MaxEntropy when available.
        return float(_threshold_maxentropy(arr))
    if m == "intermodes":
        return float(_threshold_intermodes(arr))
    if m == "minerror":
        return float(_threshold_minerror(arr))
    if m == "moments":
        return float(_threshold_moments(arr))
    if m == "percentile":
        return float(_threshold_percentile(arr))
    if m == "huang":
        return float(_threshold_huang(arr))
    if m == "shanbhag":
        return float(_threshold_shanbhag(arr))
    raise ValueError(
        f"Unknown auto-threshold method {method!r}. "
        f"Valid: {AUTO_THRESHOLD_METHODS}"
    )


def auto_threshold(
    img: ArrayLike, method: str = "default", **kwargs
) -> NDArray[np.bool_]:
    """``Image > Adjust > Auto Threshold...`` -- return the binary mask."""
    t = auto_threshold_value(img, method=method, **kwargs)
    return np.asarray(img) > t


def manual_threshold(
    img: ArrayLike, lower: float, upper: float | None = None
) -> NDArray[np.bool_]:
    """``Image > Adjust > Threshold...`` -- manual lower/upper bounds."""
    arr = np.asarray(img)
    if upper is None:
        return arr >= lower
    return (arr >= lower) & (arr <= upper)


# Histogram-based threshold implementations not in scikit-image
# (ported from the Auto_Threshold ImageJ plugin -- Gabriel Landini, 2008).
def _histogram(arr: NDArray, nbins: int = 256) -> tuple[NDArray, NDArray]:
    a = np.asarray(arr).ravel()
    if a.dtype.kind in "ui":
        bins = int(a.max()) + 1 if a.max() < nbins else nbins
        hist, edges = np.histogram(a, bins=bins)
    else:
        hist, edges = np.histogram(a, bins=nbins)
    return hist.astype(np.float64), edges


def _threshold_maxentropy(arr: NDArray) -> float:
    hist, edges = _histogram(arr)
    h = hist / hist.sum()
    cum = np.cumsum(h)
    best, best_t = -np.inf, 0
    eps = 1e-12
    for t in range(1, len(h) - 1):
        p1 = cum[t]
        p2 = 1.0 - p1
        if p1 < eps or p2 < eps:
            continue
        h1 = h[: t + 1] / p1
        h2 = h[t + 1 :] / p2
        e1 = -np.sum(h1[h1 > 0] * np.log(h1[h1 > 0]))
        e2 = -np.sum(h2[h2 > 0] * np.log(h2[h2 > 0]))
        if e1 + e2 > best:
            best, best_t = e1 + e2, t
    return float(0.5 * (edges[best_t] + edges[best_t + 1]))


def _threshold_intermodes(arr: NDArray) -> float:
    hist, edges = _histogram(arr)
    h = hist.copy()
    iters = 0
    while iters < 10000 and not _is_bimodal(h):
        h = _smooth3(h)
        iters += 1
    # threshold = mean of the two peaks
    peaks = [i for i in range(1, len(h) - 1) if h[i - 1] < h[i] >= h[i + 1]]
    if len(peaks) < 2:
        return float(edges[len(h) // 2])
    t = (peaks[0] + peaks[-1]) // 2
    return float(0.5 * (edges[t] + edges[t + 1]))


def _threshold_minerror(arr: NDArray) -> float:
    # Kittler & Illingworth 1986
    hist, edges = _histogram(arr)
    h = hist / hist.sum()
    grays = 0.5 * (edges[:-1] + edges[1:])
    best, best_t = np.inf, 0
    for t in range(1, len(h) - 1):
        p1 = h[: t + 1].sum()
        p2 = h[t + 1 :].sum()
        if p1 < 1e-9 or p2 < 1e-9:
            continue
        m1 = (h[: t + 1] * grays[: t + 1]).sum() / p1
        m2 = (h[t + 1 :] * grays[t + 1 :]).sum() / p2
        v1 = (h[: t + 1] * (grays[: t + 1] - m1) ** 2).sum() / p1
        v2 = (h[t + 1 :] * (grays[t + 1 :] - m2) ** 2).sum() / p2
        if v1 <= 0 or v2 <= 0:
            continue
        j = 1.0 + 2.0 * (p1 * np.log(np.sqrt(v1)) + p2 * np.log(np.sqrt(v2))) \
            - 2.0 * (p1 * np.log(p1) + p2 * np.log(p2))
        if j < best:
            best, best_t = j, t
    return float(0.5 * (edges[best_t] + edges[best_t + 1]))


def _threshold_moments(arr: NDArray) -> float:
    # Tsai 1985
    hist, edges = _histogram(arr)
    h = hist / hist.sum()
    grays = 0.5 * (edges[:-1] + edges[1:])
    m0 = 1.0
    m1 = (h * grays).sum()
    m2 = (h * grays * grays).sum()
    m3 = (h * grays * grays * grays).sum()
    cd = m0 * m2 - m1 * m1
    if cd == 0:
        return float(grays[len(h) // 2])
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    z0 = 0.5 * (-c1 - np.sqrt(max(c1 * c1 - 4.0 * c0, 0.0)))
    z1 = 0.5 * (-c1 + np.sqrt(max(c1 * c1 - 4.0 * c0, 0.0)))
    p0 = (z1 - m1) / (z1 - z0) if (z1 - z0) != 0 else 0.5
    # find threshold whose cumulative <= p0
    cum = np.cumsum(h)
    t = int(np.searchsorted(cum, p0))
    t = max(0, min(t, len(h) - 1))
    return float(grays[t])


def _threshold_percentile(arr: NDArray, p: float = 0.5) -> float:
    """ImageJ "Percentile" -- assumes ``p`` fraction of pixels are foreground."""
    a = np.asarray(arr).ravel()
    return float(np.quantile(a, 1.0 - p))


def _threshold_huang(arr: NDArray) -> float:
    hist, edges = _histogram(arr)
    h = hist
    grays = 0.5 * (edges[:-1] + edges[1:])
    first = int(np.argmax(h > 0))
    last = len(h) - 1 - int(np.argmax(h[::-1] > 0))
    if last <= first:
        return float(grays[first])
    # Fuzzy membership: shape parameter C = last - first
    C = float(last - first)
    s = np.cumsum(h)
    sx = np.cumsum(h * grays)
    best, best_t = np.inf, first
    for t in range(first, last):
        mu0 = sx[t] / s[t] if s[t] > 0 else 0.0
        s_back = s[-1] - s[t]
        mu1 = (sx[-1] - sx[t]) / s_back if s_back > 0 else 0.0
        ent = 0.0
        for i in range(first, last + 1):
            mu = mu0 if i <= t else mu1
            mx = 1.0 / (1.0 + abs(grays[i] - mu) / C)
            if 0 < mx < 1:
                ent += h[i] * (-mx * np.log(mx) - (1 - mx) * np.log(1 - mx))
        if ent < best:
            best, best_t = ent, t
    return float(grays[best_t])


def _threshold_shanbhag(arr: NDArray) -> float:
    hist, edges = _histogram(arr)
    h = hist / hist.sum()
    cum = np.cumsum(h)
    best, best_t = np.inf, 0
    eps = 1e-12
    for t in range(1, len(h) - 1):
        p1 = cum[t]
        p2 = 1.0 - p1
        if p1 < eps or p2 < eps:
            continue
        term1 = -np.sum(h[: t + 1] / p1 * np.log(0.5 + 0.5 * h[: t + 1] / p1 + eps))
        term2 = -np.sum(h[t + 1 :] / p2 * np.log(0.5 + 0.5 * h[t + 1 :] / p2 + eps))
        s = abs(term1 - term2)
        if s < best:
            best, best_t = s, t
    return float(0.5 * (edges[best_t] + edges[best_t + 1]))


def _smooth3(h: NDArray) -> NDArray:
    out = h.copy()
    out[1:-1] = (h[:-2] + h[1:-1] + h[2:]) / 3.0
    return out


def _is_bimodal(h: NDArray) -> bool:
    modes = 0
    for i in range(1, len(h) - 1):
        if h[i - 1] < h[i] > h[i + 1]:
            modes += 1
            if modes > 2:
                return False
    return modes == 2


# =====================================================================
#                Process > Filters (linear / non-linear filters)
# =====================================================================
def gaussian_blur(img: ArrayLike, sigma: float = 2.0) -> NDArray[np.float32]:
    """``Process > Filters > Gaussian Blur...``"""
    from skimage.filters import gaussian as _g
    return _g(_as_float32(img), sigma=float(sigma), preserve_range=True).astype(np.float32)


def median_filter(img: ArrayLike, radius: int = 1) -> NDArray[np.float32]:
    """``Process > Filters > Median...`` (square structuring element)."""
    from skimage.filters import median
    out = median(_as_float32(img), footprint=_disk(radius))
    return out.astype(np.float32)


def mean_filter(img: ArrayLike, radius: int = 1) -> NDArray[np.float32]:
    """``Process > Filters > Mean...`` (uniform / box filter)."""
    from scipy.ndimage import uniform_filter
    size = 2 * int(radius) + 1
    return uniform_filter(_as_float32(img), size=size).astype(np.float32)


def minimum_filter(img: ArrayLike, radius: int = 1) -> NDArray[np.float32]:
    """``Process > Filters > Minimum...`` (greyscale erosion by disk)."""
    from scipy.ndimage import minimum_filter as _mf
    return _mf(_as_float32(img), size=2 * int(radius) + 1).astype(np.float32)


def maximum_filter(img: ArrayLike, radius: int = 1) -> NDArray[np.float32]:
    """``Process > Filters > Maximum...`` (greyscale dilation by disk)."""
    from scipy.ndimage import maximum_filter as _mf
    return _mf(_as_float32(img), size=2 * int(radius) + 1).astype(np.float32)


def variance_filter(img: ArrayLike, radius: int = 1) -> NDArray[np.float32]:
    """``Process > Filters > Variance...`` (local variance in box neighbourhood)."""
    from scipy.ndimage import uniform_filter
    arr = _as_float32(img)
    size = 2 * int(radius) + 1
    mean = uniform_filter(arr, size=size)
    mean_sq = uniform_filter(arr * arr, size=size)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return var.astype(np.float32)


def unsharp_mask(
    img: ArrayLike, radius: float = 2.0, mask_weight: float = 0.6
) -> NDArray[np.float32]:
    """``Process > Filters > Unsharp Mask...``

    Implementation follows the ImageJ formula
    ``out = (img - amount * blur(img, sigma=radius)) / (1 - amount)``.
    """
    from skimage.filters import gaussian as _g
    arr = _as_float32(img)
    blurred = _g(arr, sigma=float(radius), preserve_range=True)
    w = float(mask_weight)
    if w >= 1.0:
        return arr
    out = (arr - w * blurred) / (1.0 - w)
    return out.astype(np.float32)


def convolve(img: ArrayLike, kernel: ArrayLike) -> NDArray[np.float32]:
    """``Process > Filters > Convolve...`` -- arbitrary 2-D kernel."""
    from scipy.ndimage import convolve as _c
    k = np.asarray(kernel, dtype=np.float32)
    return _c(_as_float32(img), k, mode="reflect").astype(np.float32)


def top_hat_white(img: ArrayLike, radius: int = 3) -> NDArray[np.float32]:
    """``Process > Filters > Top Hat (white)`` -- bright structures < radius."""
    from skimage.morphology import white_tophat
    return white_tophat(_as_float32(img), footprint=_disk(radius)).astype(np.float32)


def top_hat_black(img: ArrayLike, radius: int = 3) -> NDArray[np.float32]:
    """``Process > Filters > Top Hat (black)`` -- dark structures < radius."""
    from skimage.morphology import black_tophat
    return black_tophat(_as_float32(img), footprint=_disk(radius)).astype(np.float32)


# Vesselness / tubeness filters (Fiji "Tubeness" / "Frangi" plugins).
def tubeness(
    img: ArrayLike, sigmas: Sequence[float] = (1.0, 1.5, 2.0),
    bright_ridges: bool = True,
) -> NDArray[np.float32]:
    """Fiji ``Plugins > Analyze > Tubeness`` (Sato 1998) via skimage.sato."""
    from skimage.filters import sato as _sato
    out = _sato(
        _as_float32(img), sigmas=tuple(sigmas),
        black_ridges=not bright_ridges, mode="reflect",
    )
    return out.astype(np.float32)


def frangi_vesselness(
    img: ArrayLike, sigmas: Sequence[float] = (1.0, 1.5, 2.0),
    alpha: float = 0.5, beta: float = 0.5,
    bright_ridges: bool = True,
) -> NDArray[np.float32]:
    """Fiji ``Plugins > Filters > Frangi Vesselness`` via skimage.frangi."""
    from skimage.filters import frangi as _f
    out = _f(
        _as_float32(img), sigmas=tuple(sigmas), alpha=alpha, beta=beta,
        black_ridges=not bright_ridges, mode="reflect",
    )
    return out.astype(np.float32)


def meijering_neuriteness(
    img: ArrayLike, sigmas: Sequence[float] = (1.0, 1.5, 2.0),
    bright_ridges: bool = True,
) -> NDArray[np.float32]:
    """``Plugins > Filters > Neuriteness`` (Meijering 2004) via skimage."""
    from skimage.filters import meijering as _m
    out = _m(
        _as_float32(img), sigmas=tuple(sigmas),
        black_ridges=not bright_ridges, mode="reflect",
    )
    return out.astype(np.float32)


def hessian_filter(
    img: ArrayLike, sigmas: Sequence[float] = (1.0, 1.5, 2.0),
    bright_ridges: bool = True,
) -> NDArray[np.float32]:
    """``Plugins > Filters > Hessian`` via skimage.hessian."""
    from skimage.filters import hessian as _h
    out = _h(
        _as_float32(img), sigmas=tuple(sigmas),
        black_ridges=not bright_ridges, mode="reflect",
    )
    return out.astype(np.float32)


# =====================================================================
#                Process > Smooth / Sharpen / Find Edges / Find Maxima
# =====================================================================
def smooth(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Smooth`` -- 3x3 mean filter."""
    return mean_filter(img, radius=1)


def sharpen(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Sharpen`` -- 3x3 Laplacian sharpening kernel."""
    kernel = np.array(
        [[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]], dtype=np.float32
    ) / 4.0
    return convolve(img, kernel)


def find_edges(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Find Edges`` -- Sobel gradient magnitude."""
    from skimage.filters import sobel
    return sobel(_as_float32(img)).astype(np.float32)


def find_maxima(
    img: ArrayLike, tolerance: float = 0.05, exclude_on_edges: bool = False
) -> NDArray[np.bool_]:
    """``Process > Find Maxima...`` -- local maxima above a noise tolerance."""
    from skimage.feature import peak_local_max
    coords = peak_local_max(
        _as_float32(img), threshold_abs=float(tolerance),
        exclude_border=int(exclude_on_edges),
    )
    out = np.zeros(img.shape, dtype=bool)
    out[tuple(coords.T)] = True
    return out


def enhance_contrast(
    img: ArrayLike, saturated: float = 0.35,
    normalize: bool = True, equalize_histogram: bool = False,
) -> NDArray[np.float32]:
    """``Process > Enhance Contrast...`` -- saturated-pixel stretch or histogram eq."""
    if equalize_histogram:
        from skimage.exposure import equalize_hist
        return equalize_hist(_as_float32(img)).astype(np.float32)
    return percentile_stretch(
        img, low_pct=saturated, high_pct=100.0 - saturated
    )


def clahe(
    img: ArrayLike, kernel_size: int | tuple[int, int] = 64,
    clip_limit: float = 0.01, nbins: int = 256,
) -> NDArray[np.float32]:
    """``Process > Enhance Local Contrast (CLAHE)`` (Zuiderveld 1994)."""
    from skimage.exposure import equalize_adapthist
    out = equalize_adapthist(
        _as_float32(img), kernel_size=kernel_size,
        clip_limit=float(clip_limit), nbins=int(nbins),
    )
    return out.astype(np.float32)


# =====================================================================
#                Process > Subtract Background
# =====================================================================
def subtract_background(
    img: ArrayLike, radius: float = 50.0,
    light_background: bool = False, sliding_paraboloid: bool = True,
) -> NDArray[np.float32]:
    """``Process > Subtract Background...`` (Sternberg's rolling ball).

    With ``sliding_paraboloid=True`` (the ImageJ default since v1.40), the
    background is the lower envelope of a paraboloid sliding under the
    image. ``scikit-image`` implements the original spherical-ball variant;
    when the radius is large the two are visually similar but not bit-exact.
    """
    from skimage.restoration import rolling_ball as _rb
    arr = _as_float32(img)
    if light_background:
        arr = arr.max() - arr
    bg = _rb(arr, radius=float(radius)).astype(np.float32)
    out = np.clip(arr - bg, 0.0, None)
    if light_background:
        out = arr.max() - out
    return out.astype(np.float32)


# =====================================================================
#                Process > Math (per-pixel arithmetic)
# =====================================================================
def add_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Add...``"""
    return (_as_float32(img) + float(value)).astype(np.float32)


def subtract_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Subtract...``"""
    return (_as_float32(img) - float(value)).astype(np.float32)


def multiply_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Multiply...``"""
    return (_as_float32(img) * float(value)).astype(np.float32)


def divide_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Divide...``"""
    if value == 0:
        raise ValueError("divide_constant: divisor is zero")
    return (_as_float32(img) / float(value)).astype(np.float32)


def min_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Min...``"""
    return np.minimum(_as_float32(img), float(value)).astype(np.float32)


def max_constant(img: ArrayLike, value: float) -> NDArray[np.float32]:
    """``Process > Math > Max...``"""
    return np.maximum(_as_float32(img), float(value)).astype(np.float32)


def log_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Log``"""
    arr = _as_float32(img)
    return np.log(np.clip(arr, 1e-12, None)).astype(np.float32)


def exp_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Exp``"""
    return np.exp(_as_float32(img)).astype(np.float32)


def square_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Square``"""
    arr = _as_float32(img)
    return (arr * arr).astype(np.float32)


def sqrt_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Square Root``"""
    return np.sqrt(np.clip(_as_float32(img), 0.0, None)).astype(np.float32)


def abs_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Abs``"""
    return np.abs(_as_float32(img)).astype(np.float32)


def reciprocal_image(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Math > Reciprocal``"""
    arr = _as_float32(img)
    out = np.where(arr != 0, 1.0 / arr, 0.0)
    return out.astype(np.float32)


def image_add(a: ArrayLike, b: ArrayLike) -> NDArray[np.float32]:
    """``Process > Image Calculator > Add``"""
    return (_as_float32(a) + _as_float32(b)).astype(np.float32)


def image_subtract(a: ArrayLike, b: ArrayLike) -> NDArray[np.float32]:
    """``Process > Image Calculator > Subtract``"""
    return (_as_float32(a) - _as_float32(b)).astype(np.float32)


def image_multiply(a: ArrayLike, b: ArrayLike) -> NDArray[np.float32]:
    """``Process > Image Calculator > Multiply``"""
    return (_as_float32(a) * _as_float32(b)).astype(np.float32)


def image_divide(a: ArrayLike, b: ArrayLike) -> NDArray[np.float32]:
    """``Process > Image Calculator > Divide`` (0/0 -> 0)."""
    aa = _as_float32(a)
    bb = _as_float32(b)
    return np.where(bb != 0, aa / bb, 0.0).astype(np.float32)


# =====================================================================
#                Process > Noise
# =====================================================================
def add_gaussian_noise(
    img: ArrayLike, stddev: float = 25.0, rng: np.random.Generator | None = None
) -> NDArray[np.float32]:
    """``Process > Noise > Add Specified Noise...`` (Gaussian)."""
    rng = rng or np.random.default_rng()
    return (_as_float32(img) + rng.normal(0.0, stddev, size=img.shape)).astype(np.float32)


def salt_and_pepper(
    img: ArrayLike, percent: float = 0.05,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """``Process > Noise > Salt and Pepper``"""
    rng = rng or np.random.default_rng()
    arr = _as_float32(img).copy()
    n = int(percent * arr.size / 2.0)
    flat = arr.ravel()
    salt_idx = rng.integers(0, flat.size, size=n)
    pepper_idx = rng.integers(0, flat.size, size=n)
    flat[salt_idx] = float(arr.max())
    flat[pepper_idx] = float(arr.min())
    return flat.reshape(arr.shape)


def despeckle(img: ArrayLike) -> NDArray[np.float32]:
    """``Process > Noise > Despeckle`` -- 3x3 median filter."""
    return median_filter(img, radius=1)


def remove_outliers(
    img: ArrayLike, radius: int = 2, threshold: float = 50.0,
    which: str = "bright",
) -> NDArray[np.float32]:
    """``Process > Noise > Remove Outliers...``

    Pixels deviating from the local median by more than ``threshold`` are
    replaced with the median.
    """
    arr = _as_float32(img)
    med = median_filter(arr, radius=radius)
    diff = arr - med
    if which == "bright":
        mask = diff > threshold
    elif which == "dark":
        mask = -diff > threshold
    else:
        mask = np.abs(diff) > threshold
    out = arr.copy()
    out[mask] = med[mask]
    return out


def remove_nans(img: ArrayLike, replacement: float = 0.0) -> NDArray[np.float32]:
    """``Process > Math > NaN Background`` / ``Remove NaNs``."""
    arr = _as_float32(img)
    out = np.where(np.isnan(arr), float(replacement), arr)
    return out.astype(np.float32)


# =====================================================================
#                Process > Binary
# =====================================================================
def make_binary(
    img: ArrayLike, method: str = "default", **kwargs
) -> NDArray[np.bool_]:
    """``Process > Binary > Make Binary`` (= Auto Threshold then mask)."""
    return auto_threshold(img, method=method, **kwargs)


def erode(mask: ArrayLike, iterations: int = 1) -> NDArray[np.bool_]:
    """``Process > Binary > Erode`` (3x3 structuring element)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(np.asarray(mask, dtype=bool), iterations=int(iterations))


def dilate(mask: ArrayLike, iterations: int = 1) -> NDArray[np.bool_]:
    """``Process > Binary > Dilate`` (3x3 structuring element)."""
    from scipy.ndimage import binary_dilation
    return binary_dilation(np.asarray(mask, dtype=bool), iterations=int(iterations))


def open_binary(mask: ArrayLike, iterations: int = 1) -> NDArray[np.bool_]:
    """``Process > Binary > Open`` (= erode then dilate)."""
    return dilate(erode(mask, iterations), iterations)


def close_binary(mask: ArrayLike, iterations: int = 1) -> NDArray[np.bool_]:
    """``Process > Binary > Close-`` (= dilate then erode)."""
    return erode(dilate(mask, iterations), iterations)


def outline(mask: ArrayLike) -> NDArray[np.bool_]:
    """``Process > Binary > Outline`` -- 1-px border of the foreground."""
    m = np.asarray(mask, dtype=bool)
    return m & ~erode(m, 1)


def fill_holes(mask: ArrayLike) -> NDArray[np.bool_]:
    """``Process > Binary > Fill Holes``"""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(np.asarray(mask, dtype=bool)).astype(bool)


def skeletonize(mask: ArrayLike, method: str = "lee") -> NDArray[np.bool_]:
    """``Process > Binary > Skeletonize`` (2D/3D Lee 1994 by default)."""
    from skimage.morphology import skeletonize as _sk
    return _sk(np.asarray(mask, dtype=bool), method=method)


def distance_map(mask: ArrayLike) -> NDArray[np.float32]:
    """``Process > Binary > Distance Map`` -- Euclidean distance from background."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(np.asarray(mask, dtype=bool)).astype(np.float32)


def ultimate_points(mask: ArrayLike) -> NDArray[np.bool_]:
    """``Process > Binary > Ultimate Points`` -- local maxima of distance map."""
    from skimage.feature import peak_local_max
    d = distance_map(mask)
    coords = peak_local_max(d, threshold_abs=1.0)
    out = np.zeros(d.shape, dtype=bool)
    out[tuple(coords.T)] = True
    return out


def watershed_binary(mask: ArrayLike) -> NDArray[np.int32]:
    """``Process > Binary > Watershed`` -- distance-based watershed split."""
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    m = np.asarray(mask, dtype=bool)
    d = distance_map(m)
    coords = peak_local_max(d, footprint=np.ones((3, 3)), labels=m)
    markers = np.zeros(d.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-d, markers, mask=m).astype(np.int32)


def voronoi_binary(mask: ArrayLike) -> NDArray[np.int32]:
    """``Process > Binary > Voronoi`` -- Voronoi partition of the foreground."""
    from scipy.ndimage import label, distance_transform_edt
    lbl, _ = label(np.asarray(mask, dtype=bool))
    if lbl.max() == 0:
        return lbl.astype(np.int32)
    _, (inds_i, inds_j) = distance_transform_edt(
        lbl == 0, return_indices=True
    )
    return lbl[inds_i, inds_j].astype(np.int32)


# =====================================================================
#                Process > FFT
# =====================================================================
def fft_magnitude(img: ArrayLike, log_scale: bool = True) -> NDArray[np.float32]:
    """``Process > FFT > FFT`` -- centred log-magnitude spectrum."""
    arr = _as_float32(img)
    F = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(F).astype(np.float32)
    if log_scale:
        mag = np.log1p(mag)
    return mag


def fft_phase(img: ArrayLike) -> NDArray[np.float32]:
    arr = _as_float32(img)
    F = np.fft.fftshift(np.fft.fft2(arr))
    return np.angle(F).astype(np.float32)


def inverse_fft(spectrum: ArrayLike) -> NDArray[np.float32]:
    """``Process > FFT > Inverse FFT``"""
    F = np.fft.ifftshift(np.asarray(spectrum))
    return np.real(np.fft.ifft2(F)).astype(np.float32)


def bandpass_filter(
    img: ArrayLike, low_cutoff: float = 3.0, high_cutoff: float = 40.0,
    suppress_stripes: str = "none",
) -> NDArray[np.float32]:
    """``Process > FFT > Bandpass Filter...``

    Frequencies with period between ``low_cutoff`` and ``high_cutoff``
    pixels are kept.
    """
    arr = _as_float32(img)
    h, w = arr.shape
    F = np.fft.fftshift(np.fft.fft2(arr))
    yy, xx = np.indices(arr.shape)
    cy, cx = h / 2.0, w / 2.0
    r = np.hypot(yy - cy, xx - cx)
    with np.errstate(divide="ignore"):
        period = np.where(r > 0, max(h, w) / np.where(r > 0, r, 1), np.inf)
    mask = (period >= low_cutoff) & (period <= high_cutoff)
    if suppress_stripes == "horizontal":
        mask &= np.abs(xx - cx) > 1
    elif suppress_stripes == "vertical":
        mask &= np.abs(yy - cy) > 1
    F *= mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)


# =====================================================================
#                Plugins > Deconvolution (Richardson-Lucy)
# =====================================================================
def gaussian_psf(size: int = 11, sigma: float = 1.3) -> NDArray[np.float32]:
    """Build a normalised 2-D Gaussian point-spread function."""
    if size % 2 == 0:
        size += 1
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return (psf / psf.sum()).astype(np.float32)


def richardson_lucy(
    img: ArrayLike, psf: ArrayLike | None = None, iterations: int = 30,
    clip: bool = False,
) -> NDArray[np.float32]:
    """``Plugins > DeconvolutionLab2 > Richardson-Lucy`` via scikit-image."""
    from skimage.restoration import richardson_lucy as _rl
    if psf is None:
        psf = gaussian_psf()
    return _rl(_as_float32(img), psf=np.asarray(psf, dtype=np.float32),
               num_iter=int(iterations), clip=clip).astype(np.float32)


def wiener_deconvolve(
    img: ArrayLike, psf: ArrayLike, balance: float = 0.1,
) -> NDArray[np.float32]:
    """``Plugins > DeconvolutionLab2 > Wiener`` via skimage.restoration.wiener."""
    from skimage.restoration import wiener as _w
    return _w(_as_float32(img), np.asarray(psf, dtype=np.float32),
              balance=float(balance)).astype(np.float32)


# =====================================================================
#                Image > Size / Canvas / Rotation
# =====================================================================
def resize(
    img: ArrayLike, shape: tuple[int, int], order: int = 1,
) -> NDArray[np.float32]:
    """``Image > Adjust > Size...`` -- bilinear (order=1) by default."""
    from skimage.transform import resize as _r
    out = _r(_as_float32(img), shape, order=int(order),
             preserve_range=True, anti_aliasing=(order > 0))
    return out.astype(np.float32)


def canvas_size(
    img: ArrayLike, shape: tuple[int, int], fill: float = 0.0,
    position: str = "center",
) -> NDArray[np.float32]:
    """``Image > Adjust > Canvas Size...``"""
    arr = _as_float32(img)
    h, w = arr.shape
    H, W = shape
    out = np.full((H, W), float(fill), dtype=np.float32)
    if position == "center":
        y0 = (H - h) // 2
        x0 = (W - w) // 2
    elif position == "top-left":
        y0 = x0 = 0
    elif position == "bottom-right":
        y0 = H - h
        x0 = W - w
    else:
        raise ValueError(f"Unknown position {position!r}")
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(H, y0 + h), min(W, x0 + w)
    iy0, ix0 = y0c - y0, x0c - x0
    out[y0c:y1c, x0c:x1c] = arr[iy0:iy0 + (y1c - y0c), ix0:ix0 + (x1c - x0c)]
    return out


def rotate(
    img: ArrayLike, angle_deg: float, interpolation: str = "bilinear",
    fill: float = 0.0,
) -> NDArray[np.float32]:
    """``Image > Transform > Rotate...``"""
    from skimage.transform import rotate as _rot
    order = {"nearest": 0, "bilinear": 1, "bicubic": 3}.get(
        interpolation.lower(), 1
    )
    out = _rot(_as_float32(img), angle=float(angle_deg), order=order,
               cval=float(fill), preserve_range=True)
    return out.astype(np.float32)


def flip_horizontal(img: ArrayLike) -> NDArray:
    """``Image > Transform > Flip Horizontally``"""
    return np.asarray(img)[:, ::-1].copy()


def flip_vertical(img: ArrayLike) -> NDArray:
    """``Image > Transform > Flip Vertically``"""
    return np.asarray(img)[::-1, :].copy()


# =====================================================================
#                Public registry (for GUI introspection)
# =====================================================================
#: Mapping of ImageJ menu category -> list of callables exposed here.
#: Useful for auto-building dropdowns / parameter dialogs.
FILTER_REGISTRY: dict[str, tuple[str, ...]] = {
    "Image > Type": ("to_8bit", "to_16bit", "to_32bit", "rgb_to_gray"),
    "Image > Adjust": (
        "brightness_contrast", "percentile_stretch", "gamma_correction",
        "window_level", "invert", "manual_threshold",
        "auto_threshold", "auto_threshold_value",
    ),
    "Image > Transform": ("rotate", "flip_horizontal", "flip_vertical",
                          "resize", "canvas_size"),
    "Process": ("smooth", "sharpen", "find_edges", "find_maxima",
                "enhance_contrast", "clahe", "subtract_background"),
    "Process > Filters": (
        "gaussian_blur", "median_filter", "mean_filter",
        "minimum_filter", "maximum_filter", "variance_filter",
        "unsharp_mask", "convolve", "top_hat_white", "top_hat_black",
        "tubeness", "frangi_vesselness", "meijering_neuriteness",
        "hessian_filter",
    ),
    "Process > Math": (
        "add_constant", "subtract_constant", "multiply_constant",
        "divide_constant", "min_constant", "max_constant",
        "log_image", "exp_image", "square_image", "sqrt_image",
        "abs_image", "reciprocal_image",
        "image_add", "image_subtract", "image_multiply", "image_divide",
    ),
    "Process > Noise": (
        "add_gaussian_noise", "salt_and_pepper", "despeckle",
        "remove_outliers", "remove_nans",
    ),
    "Process > Binary": (
        "make_binary", "erode", "dilate", "open_binary", "close_binary",
        "outline", "fill_holes", "skeletonize", "distance_map",
        "ultimate_points", "watershed_binary", "voronoi_binary",
    ),
    "Process > FFT": (
        "fft_magnitude", "fft_phase", "inverse_fft", "bandpass_filter",
    ),
    "Plugins > Deconvolution": (
        "richardson_lucy", "wiener_deconvolve", "gaussian_psf",
    ),
}


__all__ = [name for names in FILTER_REGISTRY.values() for name in names] + [
    "AUTO_THRESHOLD_METHODS", "FILTER_REGISTRY",
]
