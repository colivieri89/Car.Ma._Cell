"""The version number and the user manual must agree across the project."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _version(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8-sig")
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"no version in {path}"
    return match.group(1)


def test_versions_agree():
    versions = {
        "gui.py": _version("gui.py", r'^APP_VERSION = "([^"]+)"'),
        "package": _version("mitochondrial_network_analysis/__init__.py", r'^__version__ = "([^"]+)"'),
        "pyproject.toml": _version("pyproject.toml", r'^version = "([^"]+)"'),
        "CITATION.cff": _version("CITATION.cff", r"^version: (\S+)"),
    }
    assert len(set(versions.values())) == 1, versions


def test_manual_matches_version():
    manual = ROOT / "docs" / "Car.Ma._Cell_User_Manual.pdf"
    record = ROOT / "docs" / "manual" / "manual_record.json"
    assert manual.is_file()
    built = json.loads(record.read_text())["version"]
    current = _version("gui.py", r'^APP_VERSION = "([^"]+)"')
    assert built == current, "rebuild the manual: python docs/manual/make_manual.py"


def test_launch_scripts_line_endings():
    for name in ("Install_Mac.command", "Avvia_Mac.command"):
        assert b"\r\n" not in (ROOT / name).read_bytes(), name
