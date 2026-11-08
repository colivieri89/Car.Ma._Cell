"""End-to-end checks of the paired pipeline and of the library on synthetic images."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import main  # noqa: E402
from make_synthetic_pair import BAR_PX, BAR_UM, synthetic_cell, write_pair  # noqa: E402
from mitochondrial_network_analysis import AnalysisConfig, run_pipeline  # noqa: E402


def test_shipped_examples_are_reproducible(tmp_path):
    fused, fragmented = write_pair(tmp_path)
    from PIL import Image

    for name, fresh in (("synthetic_fused.png", fused), ("synthetic_fragmented.png", fragmented)):
        shipped = np.asarray(Image.open(ROOT / "examples" / name))
        assert np.array_equal(shipped, np.asarray(Image.open(fresh))), name


def test_defaults_point_to_shipped_examples():
    assert Path(main.DEFAULT_FUSION_IMAGE).is_file()
    assert Path(main.DEFAULT_FISSION_IMAGE).is_file()
    assert main.DEFAULT_FUSION_SCALEBAR_UM == BAR_UM
    assert main.DEFAULT_FISSION_SCALEBAR_UM == BAR_UM


def test_scale_bar_detection():
    for kind, seed in (("fused", 1), ("fragmented", 2)):
        img = main._to_grayscale_fluorescence(synthetic_cell(kind, seed))
        length, bbox = main._detect_scale_bar_length_px(img)
        assert length == BAR_PX
        assert bbox is not None


@pytest.fixture(scope="module")
def pair_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("pair")
    result = main.process_pair(
        main.DEFAULT_FUSION_IMAGE,
        main.DEFAULT_FUSION_SCALEBAR_UM,
        main.DEFAULT_FISSION_IMAGE,
        main.DEFAULT_FISSION_SCALEBAR_UM,
        output_dir=out,
        verbose=False,
    )
    return result, out


def test_pair_calibration(pair_result):
    result, _ = pair_result
    assert result["fusion_bar_px"] == BAR_PX
    assert result["fission_bar_px"] == BAR_PX
    assert result["target_um_per_px"] == pytest.approx(BAR_UM / BAR_PX)


def test_pair_writes_every_output(pair_result):
    _, out = pair_result
    for name in (
        "comparison_metrics.csv",
        "comparison_metrics_bars.png",
        "comparison_step2.png",
        "comparison_step3.png",
        "comparison_step4.png",
        "comparison_step5.png",
    ):
        assert (out / name).is_file(), name
    for cell in ("fusion", "fission"):
        for name in (
            "step2_deconvolved.png",
            "step3_skeleton.png",
            "step4_network.png",
            "step5_graph.png",
            "step4_global_metrics.json",
            "step4_global_metrics.csv",
            "step4_branches.csv",
            "step4_branch_metrics.csv",
            "step4_component_metrics.csv",
            "step4_junctions.csv",
            "step4_analyze_skeleton.csv",
        ):
            assert (out / cell / name).is_file(), f"{cell}/{name}"
        metrics = json.loads((out / cell / "step4_global_metrics.json").read_text())
        assert metrics["skeleton_pixels"] > 0


def test_fused_network_is_more_connected(pair_result):
    result, _ = pair_result
    fused = result["fusion"]["global_metrics"]
    fragmented = result["fission"]["global_metrics"]
    assert fused["n_connected_components"] < fragmented["n_connected_components"]
    assert fused["connectivity_index_length"] > fragmented["connectivity_index_length"]
    assert fused["fragmentation_index_length"] < fragmented["fragmentation_index_length"]
    assert fused["network_complexity_index"] > fragmented["network_complexity_index"]
    assert fused["n_individual_mitochondria"] < fragmented["n_individual_mitochondria"]
    for m in (fused, fragmented):
        assert 0.0 <= m["connectivity_index_length"] <= 1.0
        assert m["fragmentation_index_length"] == pytest.approx(1.0 - m["connectivity_index_length"])


def test_comparison_table(pair_result):
    _, out = pair_result
    df = pd.read_csv(out / "comparison_metrics.csv")
    assert list(df.columns[:3]) == ["metric", "fusion", "fission"]
    row = df.set_index("metric").loc["n_branches"]
    assert row.iloc[2] == pytest.approx(row["fusion"] - row["fission"])


def test_excel_export(pair_result, tmp_path):
    result, _ = pair_result
    path = main.export_metrics_excel(
        tmp_path / "metrics.xlsx",
        [
            ("Fusion", result["fusion"]["global_metrics"]),
            ("Fission", result["fission"]["global_metrics"]),
        ],
    )
    df = pd.read_excel(path)
    assert list(df.columns)[1:] == ["Fusion", "Fission"]
    assert len(df) > 50


def test_deconvolution_switch(monkeypatch, tmp_path):
    img = main._to_grayscale_fluorescence(synthetic_cell("fused", 1))
    results = {}
    for flag in (True, False):
        monkeypatch.setattr(main, "USE_DECONV", flag)
        results[flag] = main.process_image(
            None, tmp_path, save=False, verbose=False,
            prepared_image=main.normalize_image(img), pixel_size_um=0.1,
        )
    with_rl = results[True]["global_metrics"]
    without = results[False]["global_metrics"]
    assert with_rl["skeleton_pixels"] > 0 and without["skeleton_pixels"] > 0
    assert with_rl != without


def test_library_pipeline(tmp_path):
    import tifffile

    img = main._to_grayscale_fluorescence(synthetic_cell("fused", 1))[:360, :]
    src = tmp_path / "cell.tif"
    tifffile.imwrite(str(src), img.astype(np.float32))
    summary = run_pipeline(
        AnalysisConfig(
            input_path=src,
            output_dir=tmp_path / "out",
            pixel_size_um=0.1,
            deconv_iterations=5,
            psf_size=11,
            psf_sigma=1.3,
        )
    )
    assert summary["n_components"] >= 1
    assert summary["n_branches"] >= 1
    assert (tmp_path / "out" / "global_metrics.json").is_file()


def test_library_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "mitochondrial_network_analysis.cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--input" in proc.stdout


@pytest.mark.parametrize("method", ["li", "otsu", "triangle", "yen", "mean", "isodata", "minimum"])
def test_every_threshold_method_runs(monkeypatch, tmp_path, method):
    # The default multiplier (4.0) is tuned for Li; other methods need about 1.
    monkeypatch.setattr(main, "THRESHOLD_METHOD", method)
    monkeypatch.setattr(main, "THRESHOLD_FACTOR", 1.0)
    img = main._to_grayscale_fluorescence(synthetic_cell("fused", 1))
    result = main.process_image(
        None, tmp_path, save=False, verbose=False,
        prepared_image=main.normalize_image(img), pixel_size_um=0.1,
    )
    assert result["global_metrics"]["skeleton_pixels"] > 0
