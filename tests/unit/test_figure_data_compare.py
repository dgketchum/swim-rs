"""``figdata.compare`` judges table content, not machine paths, stamps, or digests;
``build_figure_data.seed_output`` seeds a scratch output directory."""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "figures"
sys.path.insert(0, str(SCRIPTS))

from figdata.compare import package_differences  # noqa: E402

DIGEST = "a" * 64


def _write(d: Path, manifest: dict, table: str = "x,y\nA,0.5\n", sidecar: dict | None = None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "fig09_table.csv").write_text(table)
    (d / "fig09_metadata.json").write_text(json.dumps(sidecar or {}))
    (d / "fig_manifest.json").write_text(json.dumps(manifest))


def _manifest(root: str, utc: str, sha: str, digest: str = DIGEST, bytes_: int = 10):
    return {
        "frozen_utc": utc,
        "repo_git_sha": sha,
        "tables": {
            "fig09_table.csv": {
                "output_sha256": digest,
                "output_bytes": bytes_,
                "sources": [
                    {"path": f"{root}/5_Flux_Ensemble/results/run23/a.csv", "sha256": digest}
                ],
                "source_hashes": {"b.csv": digest},
                "cartographic": f"{root.rsplit('/', 1)[0]}/clone/data/cartographic/x.fgb",
            }
        },
    }


def test_machine_paths_stamps_and_digests_are_ignored(tmp_path):
    frozen, built = tmp_path / "frozen", tmp_path / "built"
    _write(frozen, _manifest("/data/ssd1/swim", "2026-10-15T00:00:00", "4a2074a"))
    _write(
        built,
        _manifest("/mnt/other/swim", "2026-10-16T00:00:00", "deadbeef", "b" * 64, 11),
        table="x,y\nA,0.50000000000000001\n",
    )
    assert package_differences(built, frozen) == []


def test_content_differences_are_reported(tmp_path):
    frozen, built = tmp_path / "frozen", tmp_path / "built"
    _write(frozen, _manifest("/data/ssd1/swim", "u", "s"), sidecar={"n_sites": 47})
    _write(
        built, _manifest("/data/ssd1/swim", "u", "s"), table="x,y\nA,0.6\n", sidecar={"n_sites": 46}
    )
    diffs = package_differences(built, frozen)
    assert diffs == ["fig09_metadata.json: content differs", "fig09_table.csv: content differs"]
    other = _manifest("/data/ssd1/swim", "u", "s")
    other["tables"]["fig09_table.csv"]["sources"][0]["path"] = (
        "/data/ssd1/swim/5_Flux_Ensemble/results/run22/a.csv"
    )
    _write(built, other, sidecar={"n_sites": 47})
    assert package_differences(built, frozen) == ["fig_manifest.json: content differs"]


def test_seed_output_copies_the_frozen_package_once(tmp_path):
    import build_figure_data as cli

    frozen = tmp_path / "frozen"
    _write(frozen, _manifest("/data/ssd1/swim", "u", "s"))
    (frozen / "BUILD_STATUS.md").write_text("x")
    (frozen / "superseded_1").mkdir()
    out = tmp_path / "out"
    assert cli.seed_output(out, frozen) is True
    assert sorted(p.name for p in out.iterdir()) == [
        "fig09_metadata.json",
        "fig09_table.csv",
        "fig_manifest.json",
    ]
    assert cli.seed_output(out, frozen) is False
    assert cli.seed_output(frozen, frozen) is False
    with pytest.raises(SystemExit):
        cli.seed_output(tmp_path / "other", tmp_path / "absent")
