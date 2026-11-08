"""Run the pipeline programmatically on a user-supplied image.

Usage
-----
    python examples/run_example.py path/to/step1.tif path/to/output_dir

If no arguments are provided, a synthetic test image is generated, processed
end-to-end, and saved under ``examples/output_synthetic/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

# Make the package importable when the script is run from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mitochondrial_network_analysis import AnalysisConfig, run_pipeline  # noqa: E402


def _make_synthetic_image(path: Path, size: int = 256) -> None:
    """Create a synthetic mitochondria-like image (curved bright filaments)."""
    rng = np.random.default_rng(42)
    img = np.zeros((size, size), dtype=np.float32)

    # A few curved "filaments" using sinusoids
    xs = np.arange(size)
    for k, (amp, freq, offset) in enumerate(
        [(20, 0.04, 60), (30, 0.05, 130), (15, 0.07, 200)]
    ):
        ys = (offset + amp * np.sin(freq * xs)).astype(int)
        ys = np.clip(ys, 1, size - 2)
        for x, y in zip(xs, ys):
            img[y - 1 : y + 2, x] = 1.0

    # Optical blur + Poisson noise
    from scipy.ndimage import gaussian_filter

    img = gaussian_filter(img, sigma=2.0)
    img += rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, None)

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), img.astype(np.float32))
    print(f"Synthetic test image written to {path}")


def main() -> None:
    if len(sys.argv) >= 3:
        input_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    else:
        here = Path(__file__).resolve().parent
        input_path = here / "synthetic_step1.tif"
        output_dir = here / "output_synthetic"
        if not input_path.exists():
            _make_synthetic_image(input_path)

    config = AnalysisConfig(
        input_path=input_path,
        output_dir=output_dir,
        pixel_size_um=0.1,
        deconv_iterations=20,
        psf_size=15,
        psf_sigma=2.0,
        threshold_method="otsu",
        min_object_size=15,
        min_hole_size=15,
    )

    summary = run_pipeline(config)
    print("\n--- Summary ---")
    print(f"Connected components : {summary['n_components']}")
    print(f"Branches             : {summary['n_branches']}")
    print(f"Output directory     : {output_dir}")


if __name__ == "__main__":
    main()
