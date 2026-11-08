"""Runtime dependency check with a clear, actionable error message.

The package never installs anything silently. If a required dependency is
missing, :func:`ensure_dependencies` raises an informative ``ImportError``
that tells the user exactly how to install the missing packages.
"""

from __future__ import annotations

import importlib
from typing import Iterable, List

REQUIRED = (
    "numpy",
    "scipy",
    "skimage",       # provided by scikit-image
    "pandas",
    "networkx",
    "matplotlib",
    "tifffile",
    "imageio",
)

_PIP_NAME = {
    "skimage": "scikit-image",
}


def _missing(packages: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(_PIP_NAME.get(pkg, pkg))
    return missing


def ensure_dependencies(packages: Iterable[str] = REQUIRED) -> None:
    """Verify every required dependency is importable.

    Raises
    ------
    ImportError
        If any package is missing, with installation instructions.
    """
    missing = _missing(packages)
    if not missing:
        return
    pip_args = " ".join(missing)
    raise ImportError(
        "The following required Python packages are missing: "
        f"{', '.join(missing)}.\n"
        f"Install them with:\n    pip install {pip_args}\n"
        "or run the setup script (setup_environment.sh / setup_environment.bat)."
    )
