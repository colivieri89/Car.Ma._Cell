"""Build the user manual: docs/Car.Ma._Cell_User_Manual.pdf.

Usage, from the project folder::

    .venv/bin/python docs/manual/make_manual.py [--engine PATH]

The script runs the program on the synthetic example pair, writes the
version, the settings tables, the example figures and metrics into
docs/manual/build/, compiles carmacell_manual.tex with XeLaTeX (tectonic,
latexmk or xelatex, or the engine given with --engine) and records the
version of the manual in docs/manual/manual_record.json. The screenshots in
docs/manual/figures/ come from capture_screenshots.py (macOS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
SOURCE = "carmacell_manual.tex"
PDF = ROOT / "docs" / "Car.Ma._Cell_User_Manual.pdf"
RECORD = HERE / "manual_record.json"

sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import gui  # noqa: E402


def tex(text: object) -> str:
    """Escape plain text for LaTeX."""
    out = str(text)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(a, b)
    return out


def fmt_number(v: object) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 0.01:
            return f"{v:.4g}"
        mant, exp = f"{v:.3e}".split("e")
        return f"${float(mant):.3g}\\times10^{{{int(exp)}}}$"
    return tex(v)


def settings_table(schema: tuple) -> str:
    rows = []
    for attr, label, kind, *args in schema:
        value = getattr(main, attr)
        if kind == "bool":
            shown = "on" if value else "off"
        elif kind == "tuple":
            shown = ", ".join(f"{float(v):g}" for v in value)
        elif kind == "choice":
            shown = str(value)
        else:
            shown = f"{value:g}" if isinstance(value, float) else str(value)
        rows.append(f"{tex(label)} & \\texttt{{{tex(attr)}}} & {tex(shown)} \\\\")
    return "\n".join(rows) + "\n"


def threshold_choices() -> list[str]:
    for attr, _label, kind, *args in gui._STEP3_SETTINGS:
        if attr == "THRESHOLD_METHOD":
            return [tex(c) for c in args[0]]
    return []


KEY_METRICS = (
    ("Junction nodes $N_j$", "n_junction_nodes"),
    ("Endpoints $N_e$", "n_endpoints"),
    ("Branches $N_b$", "n_branches"),
    ("Connected components $N_c$", "n_connected_components"),
    ("Networks", "n_networks"),
    ("Individual mitochondria", "n_individual_mitochondria"),
    ("Total skeleton length (µm)", "total_skeleton_length_um"),
    ("Mean branch length (µm)", "branch_length_um_mean"),
    ("MNBN (branches per network)", "mitochondrial_connectivity_index"),
    ("Connectivity index $C_L$", "connectivity_index_length"),
    ("Fragmentation index $F_L$", "fragmentation_index_length"),
    ("Network complexity index (px$^{-2}$)", "network_complexity_index"),
    ("Fractal dimension", "fractal_dimension"),
)


def run_example() -> dict:
    out = BUILD / "example_output"
    if out.exists():
        shutil.rmtree(out)
    return main.process_pair(
        main.DEFAULT_FUSION_IMAGE,
        main.DEFAULT_FUSION_SCALEBAR_UM,
        main.DEFAULT_FISSION_IMAGE,
        main.DEFAULT_FISSION_SCALEBAR_UM,
        output_dir=out,
        verbose=False,
    )


def write_generated(result: dict) -> None:
    gen = BUILD / "generated"
    figs = BUILD / "figures"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    version = gui.APP_VERSION
    today = date.today()
    (gen / "metadata.tex").write_text(
        f"\\newcommand{{\\CMCVersion}}{{{tex(version)}}}\n"
        f"\\newcommand{{\\CMCDate}}{{{today:%B %Y}}}\n",
        encoding="utf-8",
    )
    a = result["fusion"]["global_metrics"]
    b = result["fission"]["global_metrics"]
    rows = [
        f"{label} & {fmt_number(a.get(key, float('nan')))} & {fmt_number(b.get(key, float('nan')))} \\\\"
        for label, key in KEY_METRICS
    ]
    # Table rows as macros: an \input inside a tabular breaks \bottomrule.
    (gen / "tables.tex").write_text(
        "\\newcommand{\\SettingsStepTwo}{%\n" + settings_table(gui._STEP2_SETTINGS) + "}\n"
        "\\newcommand{\\SettingsStepThree}{%\n" + settings_table(gui._STEP3_SETTINGS) + "}\n"
        "\\newcommand{\\ExampleMetrics}{%\n" + "\n".join(rows) + "\n}\n"
        "\\newcommand{\\ThresholdChoices}{" + ", ".join(threshold_choices()) + "}\n",
        encoding="utf-8",
    )
    (gen / "example_values.tex").write_text(
        f"\\newcommand{{\\ExBarPx}}{{{result['fusion_bar_px']}}}\n"
        f"\\newcommand{{\\ExUmPerPx}}{{{result['target_um_per_px']:.3g}}}\n",
        encoding="utf-8",
    )

    out = Path(result["output_dir"])
    for name in ("comparison_step2.png", "comparison_step3.png", "comparison_step4.png",
                 "comparison_step5.png", "comparison_metrics_bars.png"):
        shutil.copy2(out / name, figs / name)
    shutil.copy2(main.DEFAULT_FUSION_IMAGE, figs / "input_fused.png")
    shutil.copy2(main.DEFAULT_FISSION_IMAGE, figs / "input_fragmented.png")


def find_engine(given: str | None) -> str:
    if given:
        return given
    for name in ("tectonic", "latexmk", "xelatex"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("No XeLaTeX engine found: install tectonic or a TeX distribution, or pass --engine")


def compile_tex(engine: str) -> Path:
    name = Path(engine).name
    out = BUILD / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    if name.startswith("tectonic"):
        cmds = [[engine, "--keep-logs", "--outdir", str(out), SOURCE]]
    elif name.startswith("latexmk"):
        cmds = [[engine, "-xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={out}", SOURCE]]
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", SOURCE]
        cmds = [cmd, cmd]
    for cmd in cmds:
        subprocess.run(cmd, cwd=HERE, check=True)
    return out / (Path(SOURCE).stem + ".pdf")


def write_record(pdf: Path) -> None:
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    RECORD.write_text(
        json.dumps({"version": gui.APP_VERSION, "sha256": digest}, indent=2) + "\n",
        encoding="utf-8",
    )


def main_build() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", help="path of tectonic, latexmk or xelatex")
    args = ap.parse_args()
    engine = find_engine(args.engine)
    result = run_example()
    write_generated(result)
    built = compile_tex(engine)
    shutil.copy2(built, PDF)
    write_record(PDF)
    print(f"Written {PDF.relative_to(ROOT)} (version {gui.APP_VERSION})")


if __name__ == "__main__":
    main_build()
