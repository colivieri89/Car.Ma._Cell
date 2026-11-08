"""Apple-inspired light/dark theme for the Car.Ma._Cell Tkinter GUI.

A single entry point, :func:`apply_theme`, restyles every ttk widget (through
the fully configurable ``clam`` base theme) plus the root window and classic
tk widgets (menus, ``Text``) into a clean, macOS-like appearance: flat
surfaces, generous padding, a San-Francisco/Segoe UI font stack and the
familiar system-blue accent.

Two palettes are provided - ``"light"`` and ``"dark"`` - so the GUI can offer
a runtime appearance toggle. :func:`apply_theme` returns the resolved palette
(a plain ``dict``) so the caller can colour non-ttk surfaces such as the
embedded matplotlib figures to match.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Dict


# ---------------------------------------------------------------------------
# Palettes (values approximate Apple's macOS / iOS system colours)
# ---------------------------------------------------------------------------
_LIGHT: Dict[str, str] = {
    "mode": "light",
    "bg": "#F2F2F7",            # window chrome
    "panel": "#FFFFFF",         # cards / content surfaces
    "text": "#1D1D1F",          # primary label
    "subtext": "#6E6E73",       # secondary label
    "accent": "#007AFF",        # system blue
    "accent_active": "#0063CC",
    "accent_text": "#FFFFFF",
    "border": "#D2D2D7",
    "control": "#FFFFFF",       # secondary button face
    "control_hover": "#ECECF0",
    "entry_bg": "#FFFFFF",
    "entry_border": "#C6C6C8",
    "sel_bg": "#007AFF",
    "sel_fg": "#FFFFFF",
    "fig_bg": "#FFFFFF",        # matplotlib figure background
    "ax_bg": "#FAFAFC",         # matplotlib axes / letterbox
    "title_fg": "#1D1D1F",      # matplotlib title text
    "mono_bg": "#FFFFFF",
    "mono_fg": "#1D1D1F",
    "menu_bg": "#FFFFFF",
    "menu_fg": "#1D1D1F",
    "menu_active_bg": "#007AFF",
    "menu_active_fg": "#FFFFFF",
    "tab_inactive_fg": "#6E6E73",
}

_DARK: Dict[str, str] = {
    "mode": "dark",
    "bg": "#1C1C1E",
    "panel": "#2C2C2E",
    "text": "#F5F5F7",
    "subtext": "#98989D",
    "accent": "#0A84FF",
    "accent_active": "#409CFF",
    "accent_text": "#FFFFFF",
    "border": "#3A3A3C",
    "control": "#2C2C2E",
    "control_hover": "#3A3A3C",
    "entry_bg": "#2C2C2E",
    "entry_border": "#48484A",
    "sel_bg": "#0A84FF",
    "sel_fg": "#FFFFFF",
    "fig_bg": "#1C1C1E",
    "ax_bg": "#1C1C1E",
    "title_fg": "#F5F5F7",
    "mono_bg": "#161618",
    "mono_fg": "#EAEAEA",
    "menu_bg": "#2C2C2E",
    "menu_fg": "#F5F5F7",
    "menu_active_bg": "#0A84FF",
    "menu_active_fg": "#FFFFFF",
    "tab_inactive_fg": "#98989D",
}

PALETTES: Dict[str, Dict[str, str]] = {"light": _LIGHT, "dark": _DARK}

# Preferred UI / monospace families, most "Apple" first, with graceful
# cross-platform fallbacks.
_FONT_CANDIDATES = (
    "SF Pro Text", "SF Pro Display", ".AppleSystemUIFont",
    "Helvetica Neue", "Segoe UI", "Inter", "DejaVu Sans", "Arial",
)
_MONO_CANDIDATES = (
    "SF Mono", "Menlo", "Cascadia Code", "Consolas",
    "DejaVu Sans Mono", "Courier New",
)


def _pick(root: tk.Misc, candidates, default: str) -> str:
    try:
        available = set(tkfont.families(root))
    except Exception:  # noqa: BLE001
        return default
    for name in candidates:
        if name in available:
            return name
    return default


def pick_ui_font(root: tk.Misc) -> str:
    """Return the best available Apple-like UI font family for this system."""
    return _pick(root, _FONT_CANDIDATES, "TkDefaultFont")


def pick_mono_font(root: tk.Misc) -> str:
    """Return the best available monospace family for tabular metric output."""
    return _pick(root, _MONO_CANDIDATES, "TkFixedFont")


def get_palette(mode: str) -> Dict[str, str]:
    """Return the palette dict for ``"light"`` or ``"dark"`` (default light)."""
    return PALETTES["dark"] if str(mode).lower() == "dark" else PALETTES["light"]


def apply_theme(root: tk.Misc, style: ttk.Style, mode: str) -> Dict[str, str]:
    """Restyle ``root``/``style`` to the Apple-inspired ``mode`` appearance.

    Returns the resolved palette (with the chosen ``font_family`` /
    ``mono_family`` added) so the caller can colour matplotlib figures and
    other non-ttk surfaces to match.
    """
    p = dict(get_palette(mode))
    fam = pick_ui_font(root)
    mono = pick_mono_font(root)
    p["font_family"] = fam
    p["mono_family"] = mono
    base = 10

    # ``clam`` is the only stock theme that honours full colour configuration,
    # which is required for a working light/dark toggle on every platform.
    style.theme_use("clam")

    # ---- Global defaults ----------------------------------------------------
    style.configure(
        ".",
        background=p["bg"],
        foreground=p["text"],
        fieldbackground=p["entry_bg"],
        bordercolor=p["border"],
        darkcolor=p["bg"],
        lightcolor=p["bg"],
        troughcolor=p["bg"],
        focuscolor=p["accent"],
        font=(fam, base),
    )

    # ---- Frames & labels ----------------------------------------------------
    style.configure("TFrame", background=p["bg"])
    style.configure("Card.TFrame", background=p["panel"])
    style.configure("TLabel", background=p["bg"], foreground=p["text"], font=(fam, base))
    style.configure("Sub.TLabel", background=p["bg"], foreground=p["subtext"], font=(fam, base - 1))
    style.configure("Hero.TLabel", background=p["bg"], foreground=p["text"], font=(fam, 40, "bold"))
    style.configure("Ver.TLabel", background=p["bg"], foreground=p["subtext"], font=(fam, 15))
    style.configure("H1.TLabel", background=p["bg"], foreground=p["text"], font=(fam, 15, "bold"))
    style.configure("H2.TLabel", background=p["bg"], foreground=p["text"], font=(fam, 12, "bold"))
    style.configure("Accent.TLabel", background=p["bg"], foreground=p["accent"], font=(fam, base))
    style.configure("Status.TLabel", background=p["bg"], foreground=p["subtext"], font=(fam, base - 1))

    # ---- Buttons ------------------------------------------------------------
    style.configure(
        "TButton",
        background=p["control"],
        foreground=p["text"],
        bordercolor=p["border"],
        relief="flat",
        padding=(14, 7),
        font=(fam, base),
        focusthickness=0,
    )
    style.map(
        "TButton",
        background=[("pressed", p["control_hover"]), ("active", p["control_hover"]),
                    ("disabled", p["bg"])],
        foreground=[("disabled", p["subtext"])],
        bordercolor=[("focus", p["accent"]), ("active", p["accent"])],
    )
    style.configure(
        "Accent.TButton",
        background=p["accent"],
        foreground=p["accent_text"],
        bordercolor=p["accent"],
        relief="flat",
        padding=(16, 8),
        font=(fam, base, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("pressed", p["accent_active"]), ("active", p["accent_active"]),
                    ("disabled", p["border"])],
        foreground=[("disabled", p["subtext"])],
        bordercolor=[("!disabled", p["accent"])],
    )

    # ---- Checkbutton --------------------------------------------------------
    style.configure(
        "TCheckbutton",
        background=p["bg"], foreground=p["text"], focuscolor=p["bg"], font=(fam, base),
    )
    style.map(
        "TCheckbutton",
        background=[("active", p["bg"])],
        foreground=[("disabled", p["subtext"])],
        indicatorcolor=[("selected", p["accent"]), ("!selected", p["entry_bg"])],
    )

    # ---- Text inputs --------------------------------------------------------
    for w in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(
            w,
            fieldbackground=p["entry_bg"], background=p["entry_bg"], foreground=p["text"],
            bordercolor=p["entry_border"], insertcolor=p["text"], arrowcolor=p["text"],
            padding=4,
        )
        style.map(
            w,
            bordercolor=[("focus", p["accent"])],
            fieldbackground=[("readonly", p["entry_bg"]), ("disabled", p["bg"])],
            foreground=[("disabled", p["subtext"])],
        )
    style.map(
        "TCombobox",
        foreground=[("readonly", p["text"])],
        fieldbackground=[("readonly", p["entry_bg"])],
    )

    # ---- Notebook -----------------------------------------------------------
    style.configure("TNotebook", background=p["bg"], bordercolor=p["border"],
                    tabmargins=(6, 6, 6, 0))
    style.configure(
        "TNotebook.Tab",
        background=p["bg"], foreground=p["tab_inactive_fg"], bordercolor=p["border"],
        padding=(18, 9), font=(fam, base),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", p["panel"]), ("active", p["control_hover"])],
        foreground=[("selected", p["text"])],
    )

    # ---- Misc containers ----------------------------------------------------
    style.configure("TSeparator", background=p["border"])
    style.configure("TPanedwindow", background=p["bg"])
    style.configure("Sash", sashthickness=6, gripcount=0, background=p["border"])
    style.configure("TLabelframe", background=p["bg"], foreground=p["text"], bordercolor=p["border"])
    style.configure("TLabelframe.Label", background=p["bg"], foreground=p["subtext"])

    # ---- Scrollbars ---------------------------------------------------------
    for orient in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            orient,
            background=p["control_hover"], troughcolor=p["bg"], bordercolor=p["bg"],
            arrowcolor=p["subtext"], relief="flat",
        )
        style.map(orient, background=[("active", p["subtext"])])

    # ---- Root window + classic tk widget option defaults --------------------
    root.configure(background=p["bg"])
    root.option_add("*Menu.background", p["menu_bg"])
    root.option_add("*Menu.foreground", p["menu_fg"])
    root.option_add("*Menu.activeBackground", p["menu_active_bg"])
    root.option_add("*Menu.activeForeground", p["menu_active_fg"])
    root.option_add("*Menu.relief", "flat")
    root.option_add("*Menu.borderWidth", 0)
    root.option_add("*TCombobox*Listbox.background", p["panel"])
    root.option_add("*TCombobox*Listbox.foreground", p["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["sel_bg"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["sel_fg"])

    return p
