"""``figdata.check`` re-hashes the sources a figure manifest records."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from figdata.check import check_sources, table_sources  # noqa: E402


def _write(path: Path, text: str) -> str:
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    h_rec = _write(repo / "rec.csv", "a,b\n1,2\n")
    h_lab = _write(repo / "lab.csv", "x\n3\n")
    h_pkg = _write(repo / "pkg" / "MANIFEST.json", "{}")
    h_con = _write(repo / "con.swim", "zarr")
    _write(repo / "drift.csv", "old\n")
    payload = {
        "tables": {
            "t1.csv": {
                "sources": {
                    "record": {"path": str(repo / "rec.csv"), "sha256": h_rec, "note": "n"},
                    "labelled": "lab.csv",
                    "unhashed": "whatever.csv",
                },
                "source_hashes": {"labelled": h_lab, "derived_only": "deadbeef"},
                "container_source": {"path": str(repo / "con.swim"), "sha256": h_con},
            },
            "t2.csv": {"source_package": "pkg", "source_manifest_sha256": h_pkg},
            "t3.csv": {"sources": {"stale": {"path": "drift.csv", "sha256": "0" * 64}}},
            "t4.csv": {"sources": {"gone": {"path": "absent.csv", "sha256": "0" * 64}}},
        }
    }
    path = tmp_path / "fig_manifest.json"
    path.write_text(json.dumps(payload))
    return path, repo, payload


def test_table_sources_reads_the_three_record_shapes(manifest):
    _, _, payload = manifest
    labels = [t[0] for t in table_sources(payload["tables"]["t1.csv"])]
    assert labels == ["sources.record", "container_source", "source_hashes.labelled"]
    assert table_sources(payload["tables"]["t2.csv"]) == [
        (
            "source_manifest_sha256",
            "pkg/MANIFEST.json",
            payload["tables"]["t2.csv"]["source_manifest_sha256"],
        )
    ]


def test_check_sources_reports_drift_and_absence_only(manifest):
    path, repo, _ = manifest
    n, problems = check_sources(path, repo=repo)
    assert n == 6
    assert [p.split(":")[0] for p in problems] == ["t3.csv sources.stale", "t4.csv sources.gone"]
    assert "digest changed" in problems[0] and "missing" in problems[1]
