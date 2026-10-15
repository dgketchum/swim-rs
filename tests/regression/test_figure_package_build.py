"""The figure-data builder reproduces the frozen package byte for byte.

``scripts/figures/build_figure_data.py --all`` is rerun over a copy of
``paper/data/final/figures/`` and every emitted table is compared with the frozen
one.  Only build stamps, output-directory paths, GeoPackage write metadata, and
floating-point round-off may differ (see ``figdata.compare``).  The copy is needed because two builders read the
previous package: ``fig03`` re-registers the Figure 1 provenance records from the
prior ``fig_manifest.json`` and ``fig01`` reads the prior example selection.

Skips when the frozen package or the data root is not on this machine.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "figures"
FROZEN = REPO / "paper" / "data" / "final" / "figures"
SKIP_COPY = ("superseded_*", "BUILD_STATUS.md", "__pycache__")


@pytest.mark.slow
def test_rebuilt_package_matches_frozen(tmp_path):
    if not (FROZEN / "fig_manifest.json").exists():
        pytest.skip("frozen figure package is not on this machine")
    sys.path.insert(0, str(SCRIPTS))
    try:
        from figdata import paths  # noqa: F401 - resolves the data root at import
        from figdata.compare import package_differences
    except (PermissionError, FileNotFoundError) as exc:
        pytest.skip(f"figure builder needs the data root: {exc}")

    out = tmp_path / "figures"
    shutil.copytree(FROZEN, out, ignore=shutil.ignore_patterns(*SKIP_COPY))
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_figure_data.py"), "--all", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert package_differences(out, FROZEN) == []
