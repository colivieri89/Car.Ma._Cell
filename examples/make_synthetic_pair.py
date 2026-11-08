"""Write the synthetic example pair used by the GUI defaults and the tests.

Usage
-----
    python examples/make_synthetic_pair.py [output_dir]

Two 400 x 400 px RGB images are written (red channel only, as for a
MitoTracker-like stain): ``synthetic_fused.png``, a branched network of long
tubules, and ``synthetic_fragmented.png``, many short isolated fragments.
Each image carries a 60 px scale bar in the bottom-right corner; enter
6 um as its length, which gives 0.1 um per pixel. The images are synthetic
and are meant only to try the program.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SIZE = 400
BAR_PX = 60
BAR_UM = 6.0


def _stamp(img: np.ndarray, y: float, x: float) -> None:
    yi, xi = int(round(y)), int(round(x))
    if 1 <= yi < img.shape[0] - 1 and 1 <= xi < img.shape[1] - 1:
        img[yi - 1:yi + 2, xi - 1:xi + 2] = 1.0


def synthetic_cell(kind: str, seed: int, size: int = SIZE) -> np.ndarray:
    """Return an RGB uint8 image of a fused or fragmented synthetic network."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    cy, cx = size * 0.45, size * 0.5
    if kind == "fused":
        # Long tubules leaving the perinuclear region, each with two side branches.
        for k in range(14):
            ang = 2 * np.pi * k / 14 + rng.normal(0, 0.1)
            y, x = cy + 30 * np.sin(ang), cx + 30 * np.cos(ang)
            for step in range(130):
                ang += rng.normal(0, 0.08)
                y += np.sin(ang)
                x += np.cos(ang)
                _stamp(img, y, x)
                if step in (45, 90):
                    a2, y2, x2 = ang + 0.6, y, x
                    for _ in range(40):
                        a2 += rng.normal(0, 0.08)
                        y2 += np.sin(a2)
                        x2 += np.cos(a2)
                        _stamp(img, y2, x2)
    elif kind == "fragmented":
        # Short straight fragments scattered around the nucleus.
        for _ in range(160):
            r = rng.uniform(30, 150)
            ang = rng.uniform(0, 2 * np.pi)
            y, x = cy + r * np.sin(ang), cx + r * np.cos(ang)
            d = rng.uniform(0, 2 * np.pi)
            for _ in range(int(rng.integers(3, 10))):
                y += np.sin(d)
                x += np.cos(d)
                _stamp(img, y, x)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    img = gaussian_filter(img, 1.2)
    img = img / img.max() * 0.8 + rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0.0, 1.0)
    # Scale bar in the bottom-right corner.
    img[size - 20:size - 16, size - 30 - BAR_PX:size - 30] = 1.0
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[..., 0] = (img * 255).astype(np.uint8)
    return rgb


def write_pair(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fused = out_dir / "synthetic_fused.png"
    fragmented = out_dir / "synthetic_fragmented.png"
    Image.fromarray(synthetic_cell("fused", seed=1)).save(fused)
    Image.fromarray(synthetic_cell("fragmented", seed=2)).save(fragmented)
    return fused, fragmented


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    for path in write_pair(target):
        print(f"Written {path}")
