"""Capture the screenshots of the user manual (macOS only).

Usage, from the project folder::

    .venv/bin/python docs/manual/capture_screenshots.py

The script opens the program on the synthetic example pair, runs the
analysis, and saves one picture of every tab into docs/manual/figures/ with
the macOS ``screencapture`` tool (the terminal needs the Screen Recording
permission). With pyobjc-framework-Quartz installed, only the program window
is captured, whatever lies on top of it; otherwise the screen area of the
window is captured and nothing should cover it while the script runs.
Results go to a temporary folder, so main_output/ is untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import gui  # noqa: E402

WIDTH, HEIGHT = 1440, 880


def window_id() -> int | None:
    """CGWindowID of the program window, if Quartz is available."""
    try:
        import Quartz
    except ImportError:
        return None
    info = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    best, best_area = None, 0
    for w in info:
        if w.get("kCGWindowOwnerPID") != os.getpid() or not w.get("kCGWindowIsOnscreen"):
            continue
        if not str(w.get("kCGWindowName") or "").startswith(gui.APP_NAME):
            continue
        b = w.get("kCGWindowBounds", {})
        area = b.get("Width", 0) * b.get("Height", 0)
        if area > best_area:
            best, best_area = int(w["kCGWindowNumber"]), area
    return best


def main_capture() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="carmacell_manual_"))
    main.OUTPUT_DIR = out_dir
    gui.OUTPUT_DIR = out_dir

    app = gui.MitoGUI()
    app.geometry(f"{WIDTH}x{HEIGHT}+10+40")
    app.attributes("-topmost", True)
    app.lift()
    app.focus_force()
    # Show paths relative to the program folder, as on a fresh install.
    app._fusion_label_var.set("examples/synthetic_fused.png")
    app._fission_label_var.set("examples/synthetic_fragmented.png")

    tabs = app._notebook.tabs()
    queue: list = []

    def shot(name: str) -> None:
        app.update()
        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        target = FIGURES / f"{name}.png"
        wid = window_id()
        if wid is not None:
            cmd = ["screencapture", "-x", "-o", f"-l{wid}", str(target)]
        else:
            cmd = ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(target)]
        subprocess.run(cmd, check=True)
        # Retina captures are twice the window size: keep 1800 px of width.
        from PIL import Image

        with Image.open(target) as img:
            img = img.convert("RGB")
            if img.width > 1800:
                img = img.resize((1800, round(img.height * 1800 / img.width)), Image.LANCZOS)
            img.save(target, optimize=True)
        print("captured", name)

    def select(index: int, name: str, delay: int = 1200):
        def step():
            app._notebook.select(tabs[index])
            app.after(delay, lambda: (shot(name), next_step()))
        return step

    def run_analysis():
        app._notebook.select(tabs[1])
        app._run_analysis()
        app.after(500, wait_result)

    def wait_result():
        if app._result is None:
            app.after(500, wait_result)
            return
        app.status_var.set(
            "Done. Target = 0.1000 um/px (Fusion bar = 60px, Fission bar = 60px). "
            "Output: main_output"
        )
        app.after(1500, next_step)

    def dark():
        app._set_theme("dark")
        app._notebook.select(tabs[4])
        app.after(1500, lambda: (shot("gui_dark"), app._set_theme("light"), next_step()))

    def next_step():
        if queue:
            queue.pop(0)()
        else:
            app.destroy()

    queue.extend([
        select(0, "gui_welcome"),
        select(1, "gui_step1_inputs"),
        run_analysis,
        select(1, "gui_step1_after_run"),
        select(2, "gui_step2_deconvolved"),
        select(3, "gui_step34_networks"),
        select(4, "gui_step5_graph"),
        select(5, "gui_step6_metrics"),
        select(6, "gui_metric_definitions"),
        dark,
    ])
    app.after(2000, next_step)
    app.mainloop()


if __name__ == "__main__":
    main_capture()
