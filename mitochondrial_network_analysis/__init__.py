"""Mitochondrial Network Analysis.

A Python pipeline that reproduces, as closely as possible, an ImageJ/Fiji
workflow for the analysis of mitochondrial networks acquired by
fluorescence microscopy (deconvolution + skeletonisation + graph metrics).
"""

from __future__ import annotations

from .config import AnalysisConfig
from .pipeline import run_pipeline
from . import imagej_filters  # noqa: F401  (ImageJ-compatible filter library)

__all__ = ["AnalysisConfig", "run_pipeline", "imagej_filters", "__version__"]
__version__ = "1.1.0"
