"""Graphical interface for the mitochondrial network analysis.

Run with::

    python gui.py

The window has a top toolbar (Browse / Run / Save) and a tabbed notebook:

    * **Pipeline images**  -> Step 1 (input), Step 2 (deconvolved),
                              Step 3+4 (skeleton & annotated network).
    * **Step 5 - Graph**   -> a clean node-link graph structure of each cell
                              (MitoGraph-style: edges = mitochondrial segments,
                              nodes = junctions / endpoints), drawn at the same
                              physical scale as the earlier steps.
    * **Step 6 - Metrics** -> a standalone output tab listing every
                              quantitative metric (topology, lengths,
                              branching, connectivity / fragmentation,
                              network complexity), with a comparison bar chart.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

# pyplot is used only by main.py, in the worker thread, to write the figure
# files: it must not open Tk windows outside the main thread. The figures
# shown in the window are embedded with FigureCanvasTkAgg, which does not
# depend on the pyplot backend.
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402
from PIL import Image  # noqa: E402

from theme import apply_theme  # noqa: E402

# Import main AFTER selecting the backend.
from main import (  # noqa: E402
    COLOR_SCHEMES,
    DEFAULT_COLOR_SCHEME,
    DEFAULT_FISSION_IMAGE,
    DEFAULT_FISSION_SCALEBAR_UM,
    DEFAULT_FUSION_IMAGE,
    DEFAULT_FUSION_SCALEBAR_UM,
    DEFAULT_LABEL_A,
    DEFAULT_LABEL_B,
    DISPLAY_SCALEBAR_UM,
    OUTPUT_DIR,
    TARGET_UM_PER_PX,
    _save_metric_bar_chart,
    process_pair,
)


def _open_in_file_manager(path: Path) -> None:
    """Open ``path`` in the OS file manager (cross-platform).

    ``os.startfile`` exists only on Windows; macOS uses ``open`` and Linux
    uses ``xdg-open``.
    """
    target = str(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]  # Windows only
    elif sys.platform == "darwin":
        subprocess.run(["open", target], check=False)
    else:
        subprocess.run(["xdg-open", target], check=False)


# ---------------------------------------------------------------------------
# Application identity (shown in title bar and About dialog).
# ---------------------------------------------------------------------------
APP_NAME = "Car.Ma._Cell"
APP_VERSION = "1.1.0"
APP_AUTHORS = (
    "Dr. Carlo Olivieri",
    "Dr. Mariarosaria Negri",
    "Dr. Salvatore Sessa",
)
APP_COPYRIGHT = (
    f"\u00a9 2026 {', '.join(APP_AUTHORS)}. MIT License."
)
APP_TAGLINE = "Mitochondrial network analysis - paired comparative pipeline"

_HERE = Path(__file__).resolve().parent


# Logical groups for the Step-6 metrics tab. Each entry: (label, key).
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
        "Lengths (px)",
        (
            ("Total skeleton length",      "total_skeleton_length_px"),
            ("Mean branch length",         "mean_branch_length_px"),
            ("Median branch length",       "median_branch_length_px"),
            ("Min branch length",          "min_branch_length_px"),
            ("Max branch length",          "max_branch_length_px"),
            ("Std branch length",          "std_branch_length_px"),
        ),
    ),
    (
        "Cell normalisation",
        (
            ("ROI area [px]",              "roi_area_px"),
            ("Cell area (hull) [px]",      "cell_area_px"),
            ("Mitochondrial area [px]",    "mitochondrial_area_px"),
            ("Mito / cell area fraction",  "mitochondrial_area_fraction"),
            ("Network density [1/px]",     "network_density_px_per_px2"),
            ("Normalised skeleton length", "normalised_skeleton_length"),
        ),
    ),
    (
        "Branching index",
        (
            ("Branching index [1/px^2]",   "branching_index_per_px2"),
            ("Branches density [1/px^2]",  "branches_per_px2"),
            ("Junctions / unit length",    "junctions_per_unit_length_px"),
            ("Endpoints / unit length",    "endpoints_per_unit_length_px"),
        ),
    ),
    (
        "Connectivity & Fragmentation",
        (
            ("Connectivity index (length)",  "connectivity_index_length"),
            ("Connectivity index (count)",   "connectivity_index_count"),
            ("Fragmentation index (length)", "fragmentation_index_length"),
            ("Fragmentation index (count)",  "fragmentation_index_count"),
            ("Components density [1/px^2]",  "components_per_px2"),
        ),
    ),
    (
        "Complexity",
        (
            ("Network complexity index",    "network_complexity_index"),
        ),
    ),
)

# Explanations shown in the "Metric definitions" tab. Tuples:
#   (group_name, [(metric_name, formula_or_None, description), ...])
_METRIC_HELP = (
    (
        "Topology",
        [
            ("Skeleton pixels",
             "skeleton_pixels = sum(skeleton == True)",
             "Number of foreground pixels in the 1-pixel-wide skeleton produced "
             "by Lee 1994 thinning of the cleaned binary mask."),
            ("Endpoints (N_e)",
             "endpoints = { v in graph : degree(v) == 1 }",
             "Skeleton pixels with exactly one 8-connected neighbour. They mark "
             "the tips of free filaments."),
            ("Junction nodes (N_j)",
             "junctions = { v in graph : degree(v) >= 3 }",
             "Skeleton pixels where three or more branches meet. They define the "
             "network's connectivity."),
            ("Branches / filaments (N_b)",
             "Walks of slab nodes (deg=2) connecting two special nodes "
             "(endpoint or junction); pure cycles and isolated single pixels "
             "are also counted.",
             "Each branch carries length_px = sum of edge weights along the "
             "walk (1 for orthogonal, sqrt(2) for diagonal steps)."),
            ("Connected components (N_c)",
             "Connected components of the 8-connectivity skeleton graph.",
             "Independent sub-networks. A higher count for the same skeleton "
             "length indicates a more fragmented network."),
            ("Isolated components / filaments",
             "A component is 'isolated' if it contains no junction with deg >= 3 "
             "(i.e. it is a single linear filament or a small cycle).",
             "Non-isolated filaments belong to components that contain at least "
             "one branching junction."),
        ],
    ),
    (
        "Lengths (px)",
        [
            ("Total skeleton length (L_tot)",
             "L_tot = sum_b length_px(b)  for b in branches",
             "Sum of all branch lengths. Equivalent to the geodesic length of "
             "the entire skeleton."),
            ("Mean / median / min / max / std branch length",
             "Standard descriptive statistics over { length_px(b) }.",
             "Computed only over branches with more than one pixel (single-pixel "
             "isolated points are excluded from the histogram)."),
        ],
    ),
    (
        "Cell normalisation",
        [
            ("ROI area",
             "A_ROI = H * W (px^2)",
             "Area of the analysed region. If no ROI is set, equals the full image area."),
            ("Cell area (convex hull)",
             "A_cell = | convex_hull(mask) | (px^2)",
             "Convex hull of the cleaned mitochondrial mask. Used as a robust, "
             "scale-aware proxy of the cell footprint, so that the metrics below "
             "are dimensionless w.r.t. cell size. Falls back to |mask| if the "
             "hull cannot be computed."),
            ("Mitochondrial area",
             "A_mito = sum(mask == True) (px^2)",
             "Number of foreground pixels in the cleaned binary mask."),
            ("Mito / cell area fraction",
             "f_mito = A_mito / A_cell",
             "Fraction of the cell footprint occupied by mitochondria. Dimensionless."),
            ("Network density [1/px]",
             "rho = L_tot / A_ROI",
             "Skeleton length per unit area of the analysed image (ROI). Has "
             "units of 1/px (or 1/um if a pixel size is provided)."),
            ("Normalised skeleton length",
             "L_norm = L_tot / A_cell",
             "Skeleton length per unit cell area (convex hull of the mask). "
             "Units of 1/px."),
        ],
    ),
    (
        "Branching index",
        [
            ("Branching index [1/px^2]",
             "BI = N_j / A_cell",
             "Density of branch points per unit cell area. Captures how richly "
             "the network bifurcates relative to cell size."),
            ("Branches density [1/px^2]",
             "BD = N_b / A_cell",
             "Density of filament segments per unit cell area."),
            ("Junctions per unit length",
             "j/L = N_j / L_tot",
             "How many junctions occur per pixel of skeleton: dimensionless if "
             "L_tot is in pixels. Higher means a more inter-connected mesh."),
            ("Endpoints per unit length",
             "e/L = N_e / L_tot",
             "How many free tips occur per pixel of skeleton. High values "
             "signal a fragmented or 'fluffy' network."),
        ],
    ),
    (
        "Connectivity & Fragmentation",
        [
            ("Connectivity index (length-based)",
             "C_L = L_networked / L_tot   in [0, 1]",
             "Fraction of skeleton length that belongs to non-isolated components "
             "(i.e. components containing at least one true junction). 1 = fully "
             "connected mesh, 0 = only isolated filaments. Dimensionless."),
            ("Connectivity index (count-based)",
             "C_N = N_non_isolated / N_b   in [0, 1]",
             "Same idea but counted per branch instead of per length."),
            ("Fragmentation index (length-based)",
             "F_L = 1 - C_L   in [0, 1]",
             "Complement of the length-based connectivity. 1 = totally fragmented, "
             "0 = single connected network. Dimensionless."),
            ("Fragmentation index (count-based)",
             "F_N = N_isolated_components / N_c   in [0, 1]",
             "Fraction of the connected components that contain no junction."),
            ("Components density [1/px^2]",
             "CD = N_c / A_cell",
             "Number of independent sub-networks per unit cell area. "
             "Increases with fragmentation."),
        ],
    ),
    (
        "Complexity",
        [
            ("Network complexity index",
             "NCI = (N_j / L_tot) * (L_tot / A_cell) * C_L\n"
             "    = (N_j / A_cell) * C_L",
             "Composite score combining: branching intensity (junctions per "
             "unit length), skeleton length per unit cell area, and "
             "length-based connectivity. Units of 1/px^2, in the pixels of the "
             "common grid of the pair. Higher values indicate a richer, more "
             "interconnected mitochondrial network relative to the cell size."),
        ],
    ),
    (
        "Node-degree distribution",
        [
            ("Total nodes (N_tot)",
             "N_tot = number of skeleton pixels in the graph",
             "Every skeleton pixel becomes a node of the 8-connected graph; "
             "N_tot is therefore equal to 'Skeleton pixels'."),
            ("Nodes with K branches (N_K)",
             "N_K = | { v in graph : degree(v) == K } |\n"
             "%_K  = 100 * N_K / N_tot",
             "Count of skeleton pixels having exactly K 8-connected neighbours, "
             "and the corresponding percentage of the total nodes. K = 1 are "
             "endpoints, K = 2 are slab/path nodes, K >= 3 are junctions "
             "(branch points with K incident branches). Values are reported "
             "up to the maximum degree observed."),
            ("Junction nodes (degree >= 3)",
             "N_>=3 = sum_{K >= 3} N_K   ;   %_>=3 = 100 * N_>=3 / N_tot",
             "Aggregated count and percentage of all branch points, regardless "
             "of how many branches they carry."),
        ],
    ),
)


def _format_value(v: object) -> str:
    """Pretty-print a metric value for the tabular display."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return "n/a"
        if f == 0.0:
            return "0"
        if abs(f) >= 1e4 or (abs(f) < 1e-3 and f != 0.0):
            return f"{f:.3e}"
        return f"{f:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Settings exposed in the GUI tabs. Each entry:
#   (attr_name_in_main, ui_label, kind, *kind_args)
# kinds:
#   "bool"   -> Checkbutton
#   "int"    -> Spinbox  (kind_args: from_, to, step)
#   "float"  -> Entry    (kind_args: optional format string)
#   "choice" -> Combobox (kind_args: tuple of choices)
#   "tuple"  -> Entry, parsed as comma-separated floats
#
# Defaults are READ from main.py at GUI construction time, so whatever
# values are currently set in main.py become the GUI defaults automatically.
# ---------------------------------------------------------------------------
_STEP2_SETTINGS: tuple = (
    ("USE_DECONV",        "Use Richardson-Lucy deconvolution", "bool"),
    ("PSF_SIZE",          "PSF kernel size [px, odd]",         "int",   3, 31, 2),
    ("PSF_SIGMA",         "PSF sigma [px]",                    "float"),
    ("DECONV_ITERATIONS", "RL iterations",                     "int",   1, 200, 1),
    ("DISPLAY_LOW_PCT",   "Display stretch: low percentile",   "float"),
    ("DISPLAY_HIGH_PCT",  "Display stretch: high percentile",  "float"),
    ("DISPLAY_GAMMA",     "Display gamma",                     "float"),
)

_STEP3_SETTINGS: tuple = (
    ("BG_RADIUS",                 "Rolling-ball BG radius [px]",          "int",    1, 500, 1),
    ("THRESHOLD_METHOD",          "Auto-threshold method",                "choice",
     ("li", "otsu", "triangle", "yen", "mean", "isodata", "minimum")),
    ("THRESHOLD_FACTOR",          "Threshold multiplier",                 "float"),
    ("FRANGI_SIGMAS",             "Tubeness sigmas [px, comma-sep.]",     "tuple"),
    ("NUCLEUS_FLOOR_FACTOR",      "Nucleus-floor Otsu factor",            "float"),
    ("MIN_OBJECT_SIZE",           "Remove small objects [px]",            "int",    0, 5000, 1),
    ("MIN_HOLE_SIZE",             "Fill holes smaller than [px]",         "int",    0, 5000, 1),
    ("CLOSING_RADIUS",            "Morph. closing radius [px]",           "int",    0, 20,   1),
    ("CELL_MASK_OTSU_FACTOR",     "Cell-envelope Otsu factor",            "float"),
    ("CELL_MASK_DILATION_RADIUS", "Cell-envelope dilation radius [px]",   "int",    0, 50,   1),
    ("CELL_MASK_SMOOTH_SIGMA",    "Cell-envelope smoothing sigma [px]",   "float"),
)


def _fmt_default(value: object) -> str:
    """Format a Python value for display inside an Entry widget."""
    if isinstance(value, tuple):
        return ", ".join(f"{float(v):g}" for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _parse_setting(kind: str, raw: str, current: object) -> object:
    """Parse a string from a widget back to the proper Python type."""
    raw = raw.strip()
    if kind == "int":
        return int(float(raw))
    if kind == "float":
        return float(raw)
    if kind == "tuple":
        parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
        return tuple(float(p) for p in parts)
    if kind == "choice":
        return raw
    if kind == "bool":
        return raw in ("1", "True", "true", "yes", "on")
    return raw


class _SettingsPanel(ttk.Frame):
    """A scrollable form bound to a sequence of setting tuples.

    Reads defaults from ``main`` module attributes and writes user-edited
    values back via :meth:`apply_to_main`.
    """

    def __init__(self, parent: tk.Misc, schema: tuple, title: str) -> None:
        super().__init__(parent, padding=(8, 4))
        self._schema = schema
        self._vars: dict[str, tk.Variable] = {}
        self._kinds: dict[str, str] = {}

        # Lazy import to avoid circular issues; main is already imported above.
        import main as _main  # noqa: PLC0415

        # Values written in main.py, kept for "Reset to defaults" (a run
        # overwrites the module attributes with the edited values).
        self._defaults = {entry[0]: getattr(_main, entry[0]) for entry in schema}

        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        for i, entry in enumerate(schema, start=1):
            attr, label, kind, *args = entry
            self._kinds[attr] = kind
            current = getattr(_main, attr)
            ttk.Label(self, text=label).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)

            if kind == "bool":
                var = tk.BooleanVar(value=bool(current))
                w = ttk.Checkbutton(self, variable=var)
            elif kind == "int":
                from_, to_, step = args if len(args) == 3 else (0, 1000, 1)
                var = tk.StringVar(value=str(int(current)))
                w = ttk.Spinbox(self, from_=from_, to=to_, increment=step,
                                textvariable=var, width=10)
            elif kind == "choice":
                choices = args[0]
                var = tk.StringVar(value=str(current))
                w = ttk.Combobox(self, values=list(choices), textvariable=var,
                                 state="readonly", width=14)
            else:  # float, tuple
                var = tk.StringVar(value=_fmt_default(current))
                w = ttk.Entry(self, textvariable=var, width=14)

            w.grid(row=i, column=1, sticky="w", pady=2)
            self._vars[attr] = var

        # Reset-to-defaults button: re-reads current main.py values.
        ttk.Button(self, text="Reset to defaults",
                   command=self._reset_defaults).grid(
            row=len(schema) + 1, column=0, columnspan=2,
            sticky="w", pady=(10, 0),
        )

    def _reset_defaults(self) -> None:
        for attr, var in self._vars.items():
            default = self._defaults[attr]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(default))
            elif self._kinds[attr] == "int":
                var.set(str(int(default)))
            else:
                var.set(_fmt_default(default))

    def apply_to_main(self) -> None:
        """Write the current widget values into ``main`` module attributes."""
        import main as _main  # noqa: PLC0415
        for attr, var in self._vars.items():
            raw = var.get()
            current = getattr(_main, attr)
            try:
                value = _parse_setting(self._kinds[attr], str(raw), current)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid value for '{attr}': {raw!r} ({exc})") from exc
            setattr(_main, attr, value)


# ---------------------------------------------------------------------------
# Main paired-analysis GUI
# ---------------------------------------------------------------------------


class MitoGUI(tk.Tk):
    """Tkinter front-end for the paired :data:`APP_NAME` mito-network pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}  -  {APP_TAGLINE}")
        self.geometry("1500x980")
        self.minsize(1200, 820)

        # ---- State -----------------------------------------------------
        self._fusion_path: Path | None = (
            Path(DEFAULT_FUSION_IMAGE) if DEFAULT_FUSION_IMAGE else None
        )
        self._fission_path: Path | None = (
            Path(DEFAULT_FISSION_IMAGE) if DEFAULT_FISSION_IMAGE else None
        )
        self._result: dict | None = None  # output of process_pair
        self._worker: threading.Thread | None = None
        self._panel_canvases: dict[str, FigureCanvasTkAgg] = {}
        self._panel_figures: dict[str, Figure] = {}
        self._step2_settings: _SettingsPanel | None = None
        self._step3_settings: _SettingsPanel | None = None
        self._metrics_text: tk.Text | None = None

        # ---- Zoom & area-exclusion state -------------------------------
        # "Home" axis limits captured after each render, for the Reset button.
        self._home_limits: dict[str, list] = {}
        self._home_captured: dict[str, bool] = {}
        # Per-cell list of normalised (x0, y0, x1, y1) rectangles to exclude.
        self._exclude_regions: dict[str, list] = {"fusion": [], "fission": []}
        # Rendering info for the Step-1 axes: cell_key -> (ax, extent).
        self._step1_render: dict[str, tuple] = {}
        # Active RectangleSelector widgets (kept alive by reference).
        self._rect_selectors: list = []
        self._exclude_mode_var = tk.BooleanVar(value=False)

        # User-editable label and colour-scheme variables (created here so
        # toolbar/tab builders can reference them).
        self._label_a_var = tk.StringVar(value=DEFAULT_LABEL_A)
        self._label_b_var = tk.StringVar(value=DEFAULT_LABEL_B)
        self._color_scheme_var = tk.StringVar(value=DEFAULT_COLOR_SCHEME)

        # ---- Appearance (Apple-inspired light/dark) --------------------
        self._style = ttk.Style(self)
        self._theme_mode = tk.StringVar(value="light")
        self._pal = apply_theme(self, self._style, "light")
        self._nav_toolbars: list = []
        self._menu_widgets: list = []
        self._help_text: tk.Text | None = None
        self._theme_btn: ttk.Button | None = None

        self._build_menubar()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

        # Colour the embedded matplotlib toolbars to match the theme chrome.
        self._retheme_navbars()

        # Refresh subplot titles when the user retypes a label.
        self._label_a_var.trace_add("write", lambda *_: self._refresh_titles_only())
        self._label_b_var.trace_add("write", lambda *_: self._refresh_titles_only())

        # Try to load default preview images at startup (non-fatal).
        self.after(150, self._load_input_previews)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_radiobutton(
            label="Light appearance", variable=self._theme_mode, value="light",
            command=lambda: self._set_theme("light"),
        )
        view_menu.add_radiobutton(
            label="Dark appearance", variable=self._theme_mode, value="dark",
            command=lambda: self._set_theme("dark"),
        )
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label=f"About {APP_NAME}...", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)
        self._menu_widgets = [menubar, view_menu, help_menu]

    def _show_about(self) -> None:
        authors = "\n  - ".join(("",) + APP_AUTHORS)
        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME}  -  v{APP_VERSION}\n"
            f"{APP_TAGLINE}\n\n"
            f"Authors:{authors}\n\n"
            f"{APP_COPYRIGHT}",
        )

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side="top", fill="x")

        # ---- Row 0: cell A (default = Fusion) -----------------------------
        ttk.Button(bar, text="Browse A...", command=self._browse_fusion).grid(
            row=0, column=0, padx=(0, 6), sticky="w"
        )
        self._fusion_label_var = tk.StringVar(
            value=str(self._fusion_path) if self._fusion_path else "(no file)"
        )
        ttk.Label(bar, textvariable=self._fusion_label_var, width=58, anchor="w").grid(
            row=0, column=1, sticky="we", padx=(0, 6)
        )
        ttk.Label(bar, text="Label A:").grid(row=0, column=2, padx=(6, 2))
        ttk.Entry(bar, textvariable=self._label_a_var, width=12).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Label(bar, text="Scale bar A (um):").grid(row=0, column=4, padx=(6, 2))
        self._fusion_um_var = tk.StringVar(value=str(DEFAULT_FUSION_SCALEBAR_UM))
        ttk.Entry(bar, textvariable=self._fusion_um_var, width=8).grid(
            row=0, column=5, padx=(0, 12)
        )

        # ---- Row 1: cell B (default = Fission) ----------------------------
        ttk.Button(bar, text="Browse B...", command=self._browse_fission).grid(
            row=1, column=0, padx=(0, 6), sticky="w", pady=(4, 0)
        )
        self._fission_label_var = tk.StringVar(
            value=str(self._fission_path) if self._fission_path else "(no file)"
        )
        ttk.Label(bar, textvariable=self._fission_label_var, width=58, anchor="w").grid(
            row=1, column=1, sticky="we", padx=(0, 6), pady=(4, 0)
        )
        ttk.Label(bar, text="Label B:").grid(row=1, column=2, padx=(6, 2), pady=(4, 0))
        ttk.Entry(bar, textvariable=self._label_b_var, width=12).grid(
            row=1, column=3, padx=(0, 8), pady=(4, 0)
        )
        ttk.Label(bar, text="Scale bar B (um):").grid(
            row=1, column=4, padx=(6, 2), pady=(4, 0)
        )
        self._fission_um_var = tk.StringVar(value=str(DEFAULT_FISSION_SCALEBAR_UM))
        ttk.Entry(bar, textvariable=self._fission_um_var, width=8).grid(
            row=1, column=5, padx=(0, 12), pady=(4, 0)
        )

        # ---- Row 0 right: target um/px + Run -----------------------------
        ttk.Label(bar, text="Target um/px (blank=auto):").grid(
            row=0, column=6, padx=(6, 2)
        )
        self._target_um_var = tk.StringVar(
            value="" if TARGET_UM_PER_PX is None else str(TARGET_UM_PER_PX)
        )
        ttk.Entry(bar, textvariable=self._target_um_var, width=10).grid(
            row=0, column=7, padx=(0, 12)
        )
        self._run_btn = ttk.Button(
            bar, text="Run analysis", command=self._run_analysis, style="Accent.TButton"
        )
        self._run_btn.grid(row=0, column=8, padx=(8, 4))

        # Appearance toggle (Light <-> Dark).
        self._theme_btn = ttk.Button(
            bar, text="☾  Dark", width=9, command=self._toggle_theme
        )
        self._theme_btn.grid(row=0, column=9, padx=(4, 0))

        # ---- Row 1 right: export / output actions -------------------------
        actions = ttk.Frame(bar)
        actions.grid(row=1, column=6, columnspan=4, padx=(8, 0), pady=(4, 0), sticky="e")
        self._save_pics_btn = ttk.Button(
            actions, text="Save pictures…", command=self._save_pictures,
            state="disabled",
        )
        self._save_pics_btn.pack(side="left", padx=(0, 6))
        self._save_data_btn = ttk.Button(
            actions, text="Save data…", command=self._save_data, state="disabled"
        )
        self._save_data_btn.pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open output folder", command=self._open_output).pack(
            side="left"
        )

        bar.columnconfigure(1, weight=1)

    def _build_tabs(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        # Welcome tab is the FIRST tab and the one selected at startup.
        self._tab_welcome = self._build_welcome_tab()

        la, lb = self._resolve_labels()
        self._tab_step1 = self._build_image_tab(
            "Step 1 - Inputs",
            settings_schema=_STEP2_SETTINGS,
            settings_title="Step 2 settings (deconvolution)",
            canvas_key="step1",
            settings_key="step2",
            rows=1,
            cols=2,
            subplot_titles=(f"{la} (input)", f"{lb} (input)"),
        )

        self._tab_step2 = self._build_image_tab(
            "Step 2 - Deconvolved",
            settings_schema=_STEP3_SETTINGS,
            settings_title="Step 3 settings (skeletonization)",
            canvas_key="step2",
            settings_key="step3",
            rows=1,
            cols=2,
            subplot_titles=(f"{la} (deconvolved)", f"{lb} (deconvolved)"),
        )

        self._tab_step34 = self._build_image_tab(
            "Step 3+4 - Skeletons & Networks",
            settings_schema=None,
            settings_title=None,
            canvas_key="step34",
            settings_key=None,
            rows=2,
            cols=2,
            subplot_titles=(
                f"{la} - Skeleton",
                f"{la} - Network",
                f"{lb} - Skeleton",
                f"{lb} - Network",
            ),
        )

        self._tab_step5 = self._build_image_tab(
            "Step 5 - Graph structure",
            settings_schema=None,
            settings_title=None,
            canvas_key="step5",
            settings_key=None,
            rows=1,
            cols=2,
            subplot_titles=(f"{la} - Graph", f"{lb} - Graph"),
        )

        self._tab_metrics = self._build_metrics_tab()
        self._tab_help = self._build_help_tab()

        # Ensure the Welcome tab is the one shown on startup.
        try:
            self._notebook.select(self._tab_welcome)
        except Exception:  # noqa: BLE001
            pass

    def _build_welcome_tab(self) -> ttk.Frame:
        """Splash / About tab shown at startup."""
        frame = ttk.Frame(self._notebook)
        self._notebook.add(frame, text="Welcome")

        # Center the content vertically and horizontally.
        outer = ttk.Frame(frame)
        outer.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(outer, text=APP_NAME, style="Hero.TLabel").pack(pady=(0, 4))
        ttk.Label(outer, text=f"Version {APP_VERSION}", style="Ver.TLabel").pack(
            pady=(0, 16)
        )
        ttk.Label(outer, text=APP_TAGLINE, style="Sub.TLabel").pack(pady=(0, 24))

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(outer, text="Authors", style="H2.TLabel").pack(pady=(8, 4))
        for author in APP_AUTHORS:
            ttk.Label(outer, text=author, style="TLabel").pack()

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=16)

        ttk.Label(outer, text=APP_COPYRIGHT, style="Sub.TLabel").pack(pady=(0, 4))
        ttk.Label(
            outer,
            text="Use the Help menu for the About dialog at any time.",
            style="Sub.TLabel",
        ).pack(pady=(8, 0))

        return frame

    def _build_image_tab(
        self,
        title: str,
        *,
        settings_schema,
        settings_title,
        canvas_key: str,
        settings_key,
        rows: int,
        cols: int,
        subplot_titles: tuple[str, ...],
    ) -> ttk.Frame:
        frame = ttk.Frame(self._notebook)
        self._notebook.add(frame, text=title)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # ---- Left: figure with subplots ------------------------------
        left = ttk.Frame(paned)
        paned.add(left, weight=4)

        # Controls row (zoom for every tab; area-exclusion only for Step 1).
        controls = ttk.Frame(left, padding=(2, 4))
        controls.pack(side="top", fill="x")
        ttk.Label(controls, text="Zoom:").pack(side="left", padx=(2, 4))
        ttk.Button(
            controls, text="\u2212", width=3,
            command=lambda k=canvas_key: self._zoom_axes(k, 1.25),
        ).pack(side="left", padx=1)
        ttk.Button(
            controls, text="+", width=3,
            command=lambda k=canvas_key: self._zoom_axes(k, 0.8),
        ).pack(side="left", padx=1)
        ttk.Button(
            controls, text="Reset view", width=11,
            command=lambda k=canvas_key: self._zoom_reset(k),
        ).pack(side="left", padx=(2, 8))

        if canvas_key == "step1":
            ttk.Separator(controls, orient="vertical").pack(
                side="left", fill="y", padx=6
            )
            ttk.Checkbutton(
                controls,
                text="Exclude areas (drag on image)",
                variable=self._exclude_mode_var,
                command=self._on_toggle_exclude_mode,
            ).pack(side="left", padx=(2, 6))
            ttk.Button(
                controls, text="Clear excluded areas",
                command=self._clear_exclusions,
            ).pack(side="left", padx=2)

        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self._pal["fig_bg"])
        axes = []
        for idx, sub_title in enumerate(subplot_titles, start=1):
            ax = fig.add_subplot(rows, cols, idx)
            ax.set_facecolor(self._pal["ax_bg"])
            ax.set_title(sub_title, color=self._pal["title_fg"], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        nav = NavigationToolbar2Tk(canvas, left)
        nav.update()
        self._nav_toolbars.append(nav)

        self._panel_figures[canvas_key] = fig
        self._panel_canvases[canvas_key] = canvas
        setattr(self, f"_axes_{canvas_key}", axes)

        # ---- Right: optional settings panel ---------------------------
        if settings_schema is not None:
            right = ttk.Frame(paned)
            paned.add(right, weight=1)
            panel = _SettingsPanel(right, settings_schema, settings_title or "Settings")
            panel.pack(fill="both", expand=True, padx=4, pady=8)
            if settings_key == "step2":
                self._step2_settings = panel
            elif settings_key == "step3":
                self._step3_settings = panel

        return frame

    def _build_metrics_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self._notebook)
        self._notebook.add(frame, text="Step 6 - Metrics comparison")

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left: text table
        left = ttk.Frame(paned)
        paned.add(left, weight=2)
        self._metrics_header_var = tk.StringVar(
            value=f"{DEFAULT_LABEL_A} vs {DEFAULT_LABEL_B} - global metrics"
        )
        ttk.Label(
            left,
            textvariable=self._metrics_header_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 4))
        text = tk.Text(
            left, wrap="none", font=(self._pal["mono_family"], 10),
            bg=self._pal["mono_bg"], fg=self._pal["mono_fg"],
            insertbackground=self._pal["mono_fg"],
            selectbackground=self._pal["sel_bg"], selectforeground=self._pal["sel_fg"],
            relief="flat", borderwidth=0, highlightthickness=0, padx=10, pady=8,
        )
        ysb = ttk.Scrollbar(left, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=ysb.set, state="disabled")
        text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        ysb.pack(side="left", fill="y", pady=(0, 8))
        self._metrics_text = text

        # Right: bar-chart figure (with a colour-scheme selector on top)
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        # Colour-scheme controls. Disabled until the first run produces
        # a result; selecting a scheme afterwards re-renders the bar
        # chart in place (no re-analysis needed).
        scheme_bar = ttk.Frame(right)
        scheme_bar.pack(side="top", fill="x", padx=8, pady=(8, 0))
        ttk.Label(scheme_bar, text="Bar-chart colours:").pack(side="left")
        self._scheme_cb = ttk.Combobox(
            scheme_bar,
            textvariable=self._color_scheme_var,
            values=list(COLOR_SCHEMES.keys()),
            state="disabled",
            width=10,
        )
        self._scheme_cb.pack(side="left", padx=(6, 0))
        self._scheme_cb.bind("<<ComboboxSelected>>", self._on_color_scheme_change)
        ttk.Label(
            scheme_bar,
            text="(available after the first run)",
            foreground="#888",
        ).pack(side="left", padx=(8, 0))

        fig = Figure(figsize=(7, 8), dpi=100, facecolor=self._pal["fig_bg"])
        fig.add_subplot(1, 1, 1).set_axis_off()
        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        nav = NavigationToolbar2Tk(canvas, right)
        nav.update()
        self._nav_toolbars.append(nav)
        self._panel_figures["metrics"] = fig
        self._panel_canvases["metrics"] = canvas

        return frame

    def _build_help_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self._notebook)
        self._notebook.add(frame, text="Metric definitions")

        text = tk.Text(
            frame, wrap="word", font=(self._pal["font_family"], 10),
            bg=self._pal["panel"], fg=self._pal["text"],
            insertbackground=self._pal["text"],
            selectbackground=self._pal["sel_bg"], selectforeground=self._pal["sel_fg"],
            relief="flat", borderwidth=0, highlightthickness=0, padx=12, pady=10,
        )
        self._help_text = text
        ysb = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=ysb.set)
        text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        ysb.pack(side="left", fill="y", pady=8)

        lines: list[str] = []
        for group_title, entries in _METRIC_HELP:
            lines.append(f"=== {group_title} ===")
            for entry in entries:
                name = entry[0]
                formula = entry[1] if len(entry) > 1 else None
                description = entry[2] if len(entry) > 2 else ""
                lines.append(f"\n{name}")
                if formula:
                    lines.append(f"    {formula}")
                if description:
                    lines.append(f"    {description}")
            lines.append("")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        return frame

    def _build_statusbar(self) -> None:
        ttk.Separator(self, orient="horizontal").pack(side="bottom", fill="x")
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
            padding=(10, 4),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(
            bar,
            text=APP_COPYRIGHT,
            style="Status.TLabel",
            anchor="e",
            padding=(10, 4),
        ).pack(side="right")

    # ------------------------------------------------------------------
    # Toolbar callbacks
    # ------------------------------------------------------------------

    def _resolve_labels(self) -> tuple[str, str]:
        """Return non-empty display labels (falls back to module defaults)."""
        la = (self._label_a_var.get() or "").strip() or DEFAULT_LABEL_A
        lb = (self._label_b_var.get() or "").strip() or DEFAULT_LABEL_B
        return la, lb

    def _refresh_titles_only(self) -> None:
        """Update axis titles after the user edits Label A/B, without re-imaging."""
        la, lb = self._resolve_labels()
        if hasattr(self, "_metrics_header_var"):
            self._metrics_header_var.set(f"{la} vs {lb} - global metrics")
        titles_step1 = (f"{la} (input)", f"{lb} (input)")
        titles_step2 = (f"{la} (deconvolved)", f"{lb} (deconvolved)")
        titles_step34 = (
            f"{la} - Skeleton",
            f"{la} - Network",
            f"{lb} - Skeleton",
            f"{lb} - Network",
        )
        titles_step5 = (f"{la} - Graph", f"{lb} - Graph")
        for key, titles in (
            ("step1", titles_step1),
            ("step2", titles_step2),
            ("step34", titles_step34),
            ("step5", titles_step5),
        ):
            axes = getattr(self, f"_axes_{key}", None)
            if not axes:
                continue
            for ax, t in zip(axes, titles):
                ax.set_title(t, color=self._pal["title_fg"], fontsize=10)
            self._panel_canvases[key].draw_idle()
        # Metrics tab header (if present).
        if self._metrics_text is not None and self._result is not None:
            self._populate_metrics_text(self._result)

    def _browse_fusion(self) -> None:
        path = filedialog.askopenfilename(
            title=f"Choose {self._resolve_labels()[0]} image",
            filetypes=[
                ("Image files", "*.png *.tif *.tiff *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._fusion_path = Path(path)
        self._fusion_label_var.set(str(self._fusion_path))
        self._load_input_previews()

    def _browse_fission(self) -> None:
        path = filedialog.askopenfilename(
            title=f"Choose {self._resolve_labels()[1]} image",
            filetypes=[
                ("Image files", "*.png *.tif *.tiff *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._fission_path = Path(path)
        self._fission_label_var.set(str(self._fission_path))
        self._load_input_previews()

    def _open_output(self) -> None:
        target = Path(OUTPUT_DIR)
        target.mkdir(parents=True, exist_ok=True)
        try:
            _open_in_file_manager(target)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder failed", str(exc))

    # ------------------------------------------------------------------
    # Export (pictures + single Excel workbook)
    # ------------------------------------------------------------------
    @staticmethod
    def _ordered_metric_items() -> list[tuple[str, str]]:
        """Preferred row order + human names for the Excel export, from the
        Step-4 metric groups. Keys not listed here are appended afterwards by
        :func:`main.build_metrics_dataframe`, so no data is dropped."""
        items: list[tuple[str, str]] = []
        for _group_title, entries in _METRIC_GROUPS:
            for label, key in entries:
                items.append((label, key))
        return items

    def _save_pictures(self) -> None:
        """Copy every generated figure (PNG) to a user-chosen folder."""
        if not self._result:
            messagebox.showinfo(
                "No results yet", "Run an analysis first, then save the pictures."
            )
            return
        dest = filedialog.askdirectory(title="Choose a folder to save the pictures")
        if not dest:
            return
        try:
            import main as _main  # noqa: PLC0415
            copied = _main.export_pictures(
                self._result.get("output_dir", OUTPUT_DIR), dest
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Save pictures failed", str(exc))
            return
        if not copied:
            messagebox.showwarning(
                "Nothing to save", "No picture files were found to copy."
            )
            return
        self.status_var.set(f"Saved {len(copied)} picture(s) -> {dest}")
        messagebox.showinfo(
            "Pictures saved", f"Saved {len(copied)} image file(s) to:\n{dest}"
        )

    def _save_data(self) -> None:
        """Write all metrics to a single Excel workbook (name + one col/figure)."""
        if not self._result:
            messagebox.showinfo(
                "No results yet", "Run an analysis first, then save the data."
            )
            return
        la = self._result.get("label_a", DEFAULT_LABEL_A)
        lb = self._result.get("label_b", DEFAULT_LABEL_B)
        path = filedialog.asksaveasfilename(
            title="Save metrics as an Excel workbook",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*")],
            initialfile="CarMaCell_metrics.xlsx",
        )
        if not path:
            return
        # macOS/Tk ignores ``defaultextension``; ensure the .xlsx suffix so the
        # workbook opens correctly and openpyxl writes a valid file.
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            import main as _main  # noqa: PLC0415
            columns = [
                (la, self._result["fusion"].get("global_metrics", {})),
                (lb, self._result["fission"].get("global_metrics", {})),
            ]
            _main.export_metrics_excel(
                path, columns, ordered_items=self._ordered_metric_items()
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Save data failed", str(exc))
            return
        self.status_var.set(f"Saved metrics workbook -> {path}")
        messagebox.showinfo("Data saved", f"Metrics saved to:\n{path}")

    # ------------------------------------------------------------------
    # Appearance (light / dark)
    # ------------------------------------------------------------------
    def _toggle_theme(self) -> None:
        """Flip between the light and dark appearance."""
        self._set_theme("light" if self._theme_mode.get() == "dark" else "dark")

    def _set_theme(self, mode: str) -> None:
        """Apply the requested appearance and refresh every surface."""
        mode = "dark" if str(mode).lower() == "dark" else "light"
        self._theme_mode.set(mode)
        self._pal = apply_theme(self, self._style, mode)
        self._retheme_tk_widgets()
        self._retheme_navbars()
        self._theme_matplotlib()

        # Re-render dynamic panels so per-image titles/backgrounds pick up the
        # new colours (they set colours from ``self._pal`` on each draw).
        try:
            self._load_input_previews()
        except Exception:  # noqa: BLE001
            pass
        if self._result is not None:
            try:
                self._refresh_panels(self._result)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        if self._theme_btn is not None:
            self._theme_btn.configure(text="☀  Light" if mode == "dark" else "☾  Dark")
        if getattr(self, "status_var", None) is not None:
            self.status_var.set(f"Appearance: {mode} mode.")

    def _retheme_tk_widgets(self) -> None:
        """Recolour the classic (non-ttk) tk widgets: Text panels and menus."""
        p = self._pal
        if self._metrics_text is not None:
            self._metrics_text.configure(
                bg=p["mono_bg"], fg=p["mono_fg"], insertbackground=p["mono_fg"],
                selectbackground=p["sel_bg"], selectforeground=p["sel_fg"],
            )
        if self._help_text is not None:
            prev_state = str(self._help_text.cget("state"))
            self._help_text.configure(state="normal")
            self._help_text.configure(
                bg=p["panel"], fg=p["text"], insertbackground=p["text"],
                selectbackground=p["sel_bg"], selectforeground=p["sel_fg"],
            )
            self._help_text.configure(state=prev_state)
        for menu in getattr(self, "_menu_widgets", []):
            try:
                menu.configure(
                    background=p["menu_bg"], foreground=p["menu_fg"],
                    activebackground=p["menu_active_bg"],
                    activeforeground=p["menu_active_fg"],
                )
            except Exception:  # noqa: BLE001
                pass

    def _retheme_navbars(self) -> None:
        """Colour the embedded matplotlib navigation toolbars to match chrome."""
        p = self._pal
        for nav in getattr(self, "_nav_toolbars", []):
            try:
                nav.configure(background=p["bg"])
            except Exception:  # noqa: BLE001
                pass
            for child in nav.winfo_children():
                try:
                    child.configure(background=p["bg"])
                except Exception:  # noqa: BLE001
                    pass
                try:
                    child.configure(foreground=p["subtext"])
                except Exception:  # noqa: BLE001
                    pass

    def _theme_matplotlib(self) -> None:
        """Recolour the static parts of every embedded figure and redraw."""
        p = self._pal
        for key, fig in self._panel_figures.items():
            fig.set_facecolor(p["fig_bg"])
            for ax in fig.axes:
                ax.set_facecolor(p["fig_bg"] if key == "metrics" else p["ax_bg"])
                if ax.get_title():
                    ax.title.set_color(p["title_fg"])
                for spine in ax.spines.values():
                    spine.set_color(p["border"])
                ax.tick_params(colors=p["subtext"])
            canvas = self._panel_canvases.get(key)
            if canvas is not None:
                canvas.draw_idle()

    # ------------------------------------------------------------------
    # Input preview
    # ------------------------------------------------------------------

    def _load_input_previews(self) -> None:
        """Show the raw input images in the Step 1 tab without running the pipeline.

        Each input is rendered using its own micrometres-per-pixel scale (derived
        from the scale-bar entered by the user) so Fusion and Fission share the
        SAME physical axis even though their native pixel grids differ.
        """
        axes = getattr(self, "_axes_step1", None)
        if not axes:
            return
        try:
            from mitochondrial_network_analysis.io import load_image
            import main as _main  # for _detect_scale_bar_length_px
        except Exception:  # noqa: BLE001
            return

        def _bar_um(var):
            try:
                return float(var.get())
            except (ValueError, tk.TclError):
                return 0.0

        bar_um_pair = (_bar_um(self._fusion_um_var), _bar_um(self._fission_um_var))
        la, lb = self._resolve_labels()

        self._step1_render = {}
        rendered = []  # list of (ax, img, extent) for axis equalisation
        for ax, cell_key, path, label, bar_um in zip(
            axes,
            ("fusion", "fission"),
            (self._fusion_path, self._fission_path),
            (f"{la} (input)", f"{lb} (input)"),
            bar_um_pair,
        ):
            ax.clear()
            ax.set_facecolor(self._pal["ax_bg"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, color=self._pal["title_fg"], fontsize=10)
            if not (path and Path(path).exists()):
                ax.text(0.5, 0.5, "(no image)", color=self._pal["subtext"],
                        ha="center", va="center", transform=ax.transAxes)
                continue
            try:
                img = load_image(Path(path))
                # Pick a 2-D view for scale-bar detection.
                detect_img = img if img.ndim == 2 else img[..., 0]
                um_per_px = 0.0
                if bar_um > 0:
                    try:
                        bar_px, _bbox = _main._detect_scale_bar_length_px(detect_img)
                        if bar_px > 0:
                            um_per_px = float(bar_um) / float(bar_px)
                    except Exception:  # noqa: BLE001
                        um_per_px = 0.0
                if um_per_px <= 0:
                    # Fall back to pixel coords.
                    um_per_px = 1.0
                h, w = img.shape[:2]
                extent = (0.0, w * um_per_px, h * um_per_px, 0.0)
                if img.ndim == 3:
                    ax.imshow(img, extent=extent)
                else:
                    ax.imshow(img, cmap="gray", extent=extent)
                rendered.append((ax, extent))
                self._step1_render[cell_key] = (ax, extent)
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"Load error:\n{exc}", color="red",
                        ha="center", va="center", transform=ax.transAxes)

        self._equalize_axes(rendered)
        self._draw_exclusion_patches()
        self._setup_exclusion_selectors()
        self._panel_figures["step1"].tight_layout()
        self._panel_canvases["step1"].draw_idle()
        self._home_captured["step1"] = False

    @staticmethod
    def _equalize_axes(rendered: list) -> None:
        """Set every axis in ``rendered`` to the same physical extent.

        ``rendered`` is a list of ``(ax, (x0, x1, y0_bottom, y1_top))`` tuples
        (matplotlib ``extent`` order: left, right, bottom, top — for images
        plotted top-down we use ``(0, W*s, H*s, 0)`` so y1>y0 numerically).
        """
        if not rendered:
            return
        max_w = max(ext[1] for _, ext in rendered)
        max_h = max(ext[2] for _, ext in rendered)
        for ax, _ext in rendered:
            ax.set_xlim(0, max_w)
            ax.set_ylim(max_h, 0)
            ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # Zoom controls
    # ------------------------------------------------------------------
    def _ensure_home(self, canvas_key: str) -> None:
        """Remember the current axis limits as the 'home' view (once per render)."""
        if self._home_captured.get(canvas_key):
            return
        axes = getattr(self, f"_axes_{canvas_key}", None)
        if not axes:
            return
        self._home_limits[canvas_key] = [
            (ax.get_xlim(), ax.get_ylim()) for ax in axes
        ]
        self._home_captured[canvas_key] = True

    def _zoom_axes(self, canvas_key: str, factor: float) -> None:
        """Zoom every subplot of ``canvas_key`` about its centre.

        ``factor`` < 1 zooms in, > 1 zooms out.
        """
        axes = getattr(self, f"_axes_{canvas_key}", None)
        if not axes:
            return
        self._ensure_home(canvas_key)
        for ax in axes:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            half_x = (x1 - x0) * factor / 2.0
            half_y = (y1 - y0) * factor / 2.0
            ax.set_xlim(cx - half_x, cx + half_x)
            ax.set_ylim(cy - half_y, cy + half_y)
        canvas = self._panel_canvases.get(canvas_key)
        if canvas is not None:
            canvas.draw_idle()

    def _zoom_reset(self, canvas_key: str) -> None:
        """Restore the home view captured at the last render."""
        self._ensure_home(canvas_key)
        axes = getattr(self, f"_axes_{canvas_key}", None)
        homes = self._home_limits.get(canvas_key)
        if not axes or not homes:
            return
        for ax, (xl, yl) in zip(axes, homes):
            ax.set_xlim(xl)
            ax.set_ylim(yl)
        canvas = self._panel_canvases.get(canvas_key)
        if canvas is not None:
            canvas.draw_idle()

    # ------------------------------------------------------------------
    # Area exclusion (Step 1)
    # ------------------------------------------------------------------
    def _on_toggle_exclude_mode(self) -> None:
        """Enable/disable the rectangle selectors when the checkbox changes."""
        active = bool(self._exclude_mode_var.get())
        for sel in self._rect_selectors:
            try:
                sel.set_active(active)
            except Exception:  # noqa: BLE001
                pass
        if active:
            self.status_var.set(
                "Exclude mode ON: drag a rectangle on either image to remove "
                "an area from the analysis. Uncheck to pan/zoom."
            )

    def _setup_exclusion_selectors(self) -> None:
        """Create one RectangleSelector per Step-1 axis (idempotent)."""
        # Drop any previous selectors.
        for sel in self._rect_selectors:
            try:
                sel.set_active(False)
                sel.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        self._rect_selectors = []

        active = bool(self._exclude_mode_var.get())
        for cell_key, (ax, _extent) in self._step1_render.items():
            selector = RectangleSelector(
                ax,
                lambda eclick, erelease, k=cell_key: self._on_exclude_select(
                    k, eclick, erelease
                ),
                useblit=False,
                button=[1],  # left mouse button only
                minspanx=3,
                minspany=3,
                spancoords="pixels",
                interactive=False,
                props=dict(
                    facecolor="red", edgecolor="red",
                    alpha=0.25, fill=True, linewidth=1.2,
                ),
            )
            selector.set_active(active)
            self._rect_selectors.append(selector)

    def _on_exclude_select(self, cell_key: str, eclick, erelease) -> None:
        """Store a drawn rectangle (converted to normalised image coords)."""
        info = self._step1_render.get(cell_key)
        if info is None:
            return
        _ax, extent = info
        x_max = float(extent[1])  # W * um_per_px
        y_max = float(extent[2])  # H * um_per_px
        if x_max <= 0 or y_max <= 0:
            return
        x0, x1 = sorted((float(eclick.xdata or 0), float(erelease.xdata or 0)))
        y0, y1 = sorted((float(eclick.ydata or 0), float(erelease.ydata or 0)))
        # Normalise to [0, 1] and clamp.
        nx0 = min(max(x0 / x_max, 0.0), 1.0)
        nx1 = min(max(x1 / x_max, 0.0), 1.0)
        ny0 = min(max(y0 / y_max, 0.0), 1.0)
        ny1 = min(max(y1 / y_max, 0.0), 1.0)
        if nx1 - nx0 < 1e-3 or ny1 - ny0 < 1e-3:
            return  # ignore accidental clicks
        self._exclude_regions[cell_key].append((nx0, ny0, nx1, ny1))
        self._draw_exclusion_patches()
        self._panel_canvases["step1"].draw_idle()
        n = len(self._exclude_regions[cell_key])
        self.status_var.set(
            f"Added excluded area to {cell_key} (total: {n}). "
            f"These regions are ignored when you press Run."
        )

    def _draw_exclusion_patches(self) -> None:
        """Redraw the red overlay rectangles for every Step-1 axis."""
        for cell_key, (ax, extent) in self._step1_render.items():
            # Remove previously drawn exclusion patches.
            for patch in list(getattr(ax, "_exclude_patches", [])):
                try:
                    patch.remove()
                except Exception:  # noqa: BLE001
                    pass
            ax._exclude_patches = []  # type: ignore[attr-defined]
            x_max = float(extent[1])
            y_max = float(extent[2])
            for (nx0, ny0, nx1, ny1) in self._exclude_regions.get(cell_key, []):
                rx = nx0 * x_max
                ry = ny0 * y_max
                rw = (nx1 - nx0) * x_max
                rh = (ny1 - ny0) * y_max
                patch = Rectangle(
                    (rx, ry), rw, rh,
                    facecolor="red", edgecolor="red",
                    alpha=0.28, linewidth=1.2,
                )
                ax.add_patch(patch)
                ax._exclude_patches.append(patch)  # type: ignore[attr-defined]

    def _clear_exclusions(self) -> None:
        """Remove all excluded areas for both cells."""
        self._exclude_regions = {"fusion": [], "fission": []}
        self._draw_exclusion_patches()
        if "step1" in self._panel_canvases:
            self._panel_canvases["step1"].draw_idle()
        self.status_var.set("Cleared all excluded areas.")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    def _run_analysis(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis already running.")
            return
        la, lb = self._resolve_labels()
        if not self._fusion_path or not Path(self._fusion_path).exists():
            messagebox.showerror("Missing input", f"Please choose a valid {la} image.")
            return
        if not self._fission_path or not Path(self._fission_path).exists():
            messagebox.showerror("Missing input", f"Please choose a valid {lb} image.")
            return
        try:
            fusion_um = float(self._fusion_um_var.get())
            fission_um = float(self._fission_um_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input", "Scale bar values must be numbers (in microns)."
            )
            return
        target_str = self._target_um_var.get().strip()
        target_um: float | None
        if target_str == "":
            target_um = None
        else:
            try:
                target_um = float(target_str)
            except ValueError:
                messagebox.showerror(
                    "Invalid input", "Target um/px must be a number (or blank)."
                )
                return

        # Apply step2/step3 settings panels into the main module before running
        try:
            if self._step2_settings is not None:
                self._step2_settings.apply_to_main()
            if self._step3_settings is not None:
                self._step3_settings.apply_to_main()
        except ValueError as exc:
            messagebox.showerror("Invalid setting", str(exc))
            return

        self._run_btn.configure(state="disabled")
        self.status_var.set("Running paired analysis... (this can take several minutes)")
        color_scheme = self._color_scheme_var.get() or DEFAULT_COLOR_SCHEME
        # Snapshot the excluded areas so the worker uses a stable copy.
        exclude_a = list(self._exclude_regions.get("fusion", []))
        exclude_b = list(self._exclude_regions.get("fission", []))
        self._worker = threading.Thread(
            target=self._worker_run,
            args=(
                Path(self._fusion_path),
                fusion_um,
                Path(self._fission_path),
                fission_um,
                target_um,
                la,
                lb,
                color_scheme,
                exclude_a,
                exclude_b,
            ),
            daemon=True,
        )
        self._worker.start()

    def _worker_run(
        self,
        fusion_path: Path,
        fusion_um: float,
        fission_path: Path,
        fission_um: float,
        target_um: float | None,
        label_a: str,
        label_b: str,
        color_scheme: str,
        exclude_regions_a: list,
        exclude_regions_b: list,
    ) -> None:
        try:
            result = process_pair(
                fusion_path,
                fusion_um,
                fission_path,
                fission_um,
                Path(OUTPUT_DIR),
                target_um_per_px=target_um,
                display_scalebar_um=DISPLAY_SCALEBAR_UM,
                verbose=False,
                label_a=label_a,
                label_b=label_b,
                color_scheme=color_scheme,
                exclude_regions_a=exclude_regions_a,
                exclude_regions_b=exclude_regions_b,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.after(0, lambda exc=exc: self._on_worker_error(exc))
            return
        self.after(0, lambda: self._on_worker_done(result))

    def _on_worker_error(self, exc: BaseException) -> None:
        self._run_btn.configure(state="normal")
        self.status_var.set(f"Error: {exc}")
        messagebox.showerror("Analysis failed", f"{type(exc).__name__}: {exc}")

    def _on_worker_done(self, result: dict) -> None:
        self._result = result
        self._run_btn.configure(state="normal")
        # Enable the export buttons now that there are results to save.
        if getattr(self, "_save_pics_btn", None) is not None:
            self._save_pics_btn.configure(state="normal")
        if getattr(self, "_save_data_btn", None) is not None:
            self._save_data_btn.configure(state="normal")
        # Enable post-run colour-scheme selector and sync it with the scheme
        # actually used by the run.
        if getattr(self, "_scheme_cb", None) is not None:
            self._scheme_cb.configure(state="readonly")
        scheme_used = result.get("color_scheme", DEFAULT_COLOR_SCHEME)
        if scheme_used in COLOR_SCHEMES:
            self._color_scheme_var.set(scheme_used)
        target_um = result.get("target_um_per_px", float("nan"))
        bar_f = result.get("fusion_bar_px", 0)
        bar_s = result.get("fission_bar_px", 0)
        la = result.get("label_a", DEFAULT_LABEL_A)
        lb = result.get("label_b", DEFAULT_LABEL_B)
        self.status_var.set(
            f"Done. Target = {target_um:.4f} um/px "
            f"({la} bar = {bar_f}px, {lb} bar = {bar_s}px). "
            f"Output: {result.get('output_dir')}"
        )
        try:
            self._refresh_panels(result)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    def _on_color_scheme_change(self, _event=None) -> None:
        """Re-render the bar chart with the newly selected colour scheme.

        Only active after a successful run, since it needs the metric
        dictionaries returned by :func:`process_pair`.
        """
        result = self._result
        if not result:
            return
        scheme = self._color_scheme_var.get() or DEFAULT_COLOR_SCHEME
        if scheme not in COLOR_SCHEMES:
            scheme = DEFAULT_COLOR_SCHEME
        colors = COLOR_SCHEMES[scheme]
        out_dir = Path(result.get("output_dir", OUTPUT_DIR))
        out_dir.mkdir(parents=True, exist_ok=True)
        bars_png = out_dir / "comparison_metrics_bars.png"
        try:
            _save_metric_bar_chart(
                result["fusion"].get("global_metrics", {}),
                result["fission"].get("global_metrics", {}),
                bars_png,
                label_a=result.get("label_a", DEFAULT_LABEL_A),
                label_b=result.get("label_b", DEFAULT_LABEL_B),
                colors=colors,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Recolor failed", str(exc))
            return
        result["color_scheme"] = scheme
        result["colors"] = colors
        self._populate_metrics_chart(result)
        self.status_var.set(
            f"Bar chart recoloured with scheme '{scheme}'. Saved -> {bars_png}"
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _refresh_panels(self, result: dict) -> None:
        fusion = result["fusion"]
        fission = result["fission"]
        la = result.get("label_a", DEFAULT_LABEL_A)
        lb = result.get("label_b", DEFAULT_LABEL_B)

        # Both processed pipelines share target_um_per_px after rescaling.
        px_f = float(fusion.get("pixel_size_um") or result.get("target_um_per_px") or 1.0)
        px_s = float(fission.get("pixel_size_um") or result.get("target_um_per_px") or 1.0)

        # Step 1: raw inputs (already shown but ensure refresh from disk)
        self._load_input_previews()

        # Step 2: deconvolved images at common physical scale
        self._show_pair(
            "step2",
            fusion.get("deconvolved_disp"),
            fission.get("deconvolved_disp"),
            (f"{la} (deconvolved)", f"{lb} (deconvolved)"),
            cmap="gray",
            px_um_f=px_f,
            px_um_s=px_s,
        )

        # Step 3+4: 2x2 grid (rows = cell, cols = skeleton / network)
        # The Network panels use the pixel-registered overlay (same grid as the
        # skeleton) rather than the rendered figure raster, so Step 3+4 shares
        # the exact physical scale of the Step 2 images.
        axes = self._axes_step34
        items = [
            (axes[0], fusion.get("step3_rgb"), f"{la} - Skeleton", px_f),
            (axes[1], fusion.get("step4_overlay_rgb"), f"{la} - Network", px_f),
            (axes[2], fission.get("step3_rgb"), f"{lb} - Skeleton", px_s),
            (axes[3], fission.get("step4_overlay_rgb"), f"{lb} - Network", px_s),
        ]
        rendered: list = []
        for ax, img, title, px in items:
            ax.clear()
            ax.set_facecolor(self._pal["ax_bg"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, color=self._pal["title_fg"], fontsize=10)
            if img is None:
                ax.text(0.5, 0.5, "(no image)", color=self._pal["subtext"],
                        ha="center", va="center", transform=ax.transAxes)
                continue
            h, w = img.shape[:2]
            extent = (0.0, w * px, h * px, 0.0)
            if img.ndim == 3:
                ax.imshow(img, extent=extent)
            else:
                ax.imshow(img, cmap="gray", extent=extent)
            rendered.append((ax, extent))
        self._equalize_axes(rendered)
        self._panel_figures["step34"].tight_layout()
        self._panel_canvases["step34"].draw_idle()

        # Step 5: clean node-link graph structure (drawn at the same physical
        # scale as the other steps, so it is comparable across cells).
        import main as _main  # noqa: PLC0415
        axes5 = self._axes_step5
        g_items = [
            (axes5[0], fusion.get("graph_structure"), f"{la} - Graph", px_f),
            (axes5[1], fission.get("graph_structure"), f"{lb} - Graph", px_s),
        ]
        rendered5: list = []
        for ax, payload, title, px in g_items:
            if payload is None:
                ax.clear()
                ax.set_facecolor(self._pal["fig_bg"])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, color=self._pal["title_fg"], fontsize=10)
                ax.text(0.5, 0.5, "(no graph)", color=self._pal["subtext"],
                        ha="center", va="center", transform=ax.transAxes)
                continue
            ext = _main.draw_graph_structure(
                ax, payload, px=px,
                edge_color=self._pal["subtext"],
                bg=self._pal["fig_bg"],
                title=title, title_color=self._pal["title_fg"],
            )
            rendered5.append((ax, ext))
        self._equalize_axes(rendered5)
        self._panel_figures["step5"].tight_layout()
        self._panel_canvases["step5"].draw_idle()

        # Invalidate the stored "home" zoom limits so a later Reset view uses
        # the new image extents (step1 is reset in _load_input_previews).
        for _k in ("step2", "step34", "step5"):
            self._home_captured[_k] = False

        # Metrics text + bar chart
        self._populate_metrics_text(result)
        self._populate_metrics_chart(result)

    def _show_pair(
        self,
        key,
        img_f,
        img_s,
        titles,
        *,
        cmap=None,
        px_um_f: float = 1.0,
        px_um_s: float = 1.0,
    ) -> None:
        axes = getattr(self, f"_axes_{key}")
        rendered: list = []
        for ax, img, title, px in zip(
            axes, (img_f, img_s), titles, (px_um_f, px_um_s)
        ):
            ax.clear()
            ax.set_facecolor(self._pal["ax_bg"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, color=self._pal["title_fg"], fontsize=10)
            if img is None:
                ax.text(0.5, 0.5, "(missing)", color=self._pal["subtext"],
                        ha="center", va="center", transform=ax.transAxes)
                continue
            h, w = img.shape[:2]
            extent = (0.0, w * px, h * px, 0.0)
            if img.ndim == 3:
                ax.imshow(img, extent=extent)
            else:
                ax.imshow(img, cmap=cmap or "gray", extent=extent)
            rendered.append((ax, extent))
        self._equalize_axes(rendered)
        self._panel_figures[key].tight_layout()
        self._panel_canvases[key].draw_idle()

    def _populate_metrics_text(self, result: dict) -> None:
        text = self._metrics_text
        if text is None:
            return
        fusion_gm = result["fusion"].get("global_metrics", {})
        fission_gm = result["fission"].get("global_metrics", {})
        la = result.get("label_a", DEFAULT_LABEL_A)
        lb = result.get("label_b", DEFAULT_LABEL_B)
        if hasattr(self, "_metrics_header_var"):
            self._metrics_header_var.set(f"{la} vs {lb} - global metrics")

        # Union of all keys across both, preserving _METRIC_GROUPS order then extras.
        seen: set[str] = set()
        lines: list[str] = []
        header = f"{'Metric':<42}  {la:>16}  {lb:>16}"
        lines.append(header)
        lines.append("-" * len(header))
        for group_title, entries in _METRIC_GROUPS:
            lines.append("")
            lines.append(f"[ {group_title} ]")
            for label, key in entries:
                if key not in fusion_gm and key not in fission_gm:
                    continue
                seen.add(key)
                fv = _format_value(fusion_gm.get(key))
                sv = _format_value(fission_gm.get(key))
                lines.append(f"{label:<42}  {fv:>16}  {sv:>16}")
        extras = [
            k for k in (list(fusion_gm.keys()) + list(fission_gm.keys()))
            if k not in seen
        ]
        if extras:
            lines.append("")
            lines.append("[ Other ]")
            for key in sorted(set(extras)):
                fv = _format_value(fusion_gm.get(key))
                sv = _format_value(fission_gm.get(key))
                lines.append(f"{key:<42}  {fv:>16}  {sv:>16}")

        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    def _populate_metrics_chart(self, result: dict) -> None:
        """Render the headline-metric bar chart natively (crisp + theme-aware)."""
        import main as _main  # noqa: PLC0415
        fig = self._panel_figures["metrics"]
        canvas = self._panel_canvases["metrics"]

        fusion_gm = result["fusion"].get("global_metrics", {})
        fission_gm = result["fission"].get("global_metrics", {})
        la = result.get("label_a", DEFAULT_LABEL_A)
        lb = result.get("label_b", DEFAULT_LABEL_B)
        scheme = result.get("color_scheme", DEFAULT_COLOR_SCHEME)
        colors = COLOR_SCHEMES.get(scheme, COLOR_SCHEMES[DEFAULT_COLOR_SCHEME])
        p = self._pal
        _main.render_metric_bars(
            fig, fusion_gm, fission_gm,
            label_a=la, label_b=lb, colors=colors,
            text_color=p["title_fg"], subtext_color=p["subtext"],
            grid_color=p["border"], face=p["fig_bg"],
            suptitle=f"{la} vs {lb} — key metrics",
        )
        canvas.draw_idle()


def _save_rgb_png(rgb_float: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) float array in [0, 1] to a PNG."""
    arr = np.clip(rgb_float, 0.0, 1.0)
    arr8 = (arr * 255).astype(np.uint8)
    Image.fromarray(arr8, mode="RGB").save(str(path))


def main() -> None:
    app = MitoGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
