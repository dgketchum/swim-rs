"""Unit tests for the verification layer of the E1 benchmark evidence rebuild.

Covers ``check_gboot`` (G-BOOT derived from byte-comparing the two bootstrap
runs) and ``compare_to_pinned`` (the G-VALUES comparison core behind
``--verify``): headline values, scalar blocks, nested per-series support
counts, the expected series list, pinned input hashes, output hashes, and
temporal artifact hashes. Synthetic inputs only — no project data required.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "5_Flux_Ensemble" / "rebuild_e1_benchmark_evidence.py"


@pytest.fixture(scope="module")
def reb():
    spec = importlib.util.spec_from_file_location("rebuild_e1_benchmark_evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- check_gboot


def _boot_dirs(tmp_path):
    a = tmp_path / "canonical"
    b = tmp_path / "replicate"
    a.mkdir()
    b.mkdir()
    for name, text in [("split_daily.csv", "x,1\ny,2\n"), ("deltas.csv", "d,0.1\n")]:
        (a / name).write_text(text)
        (b / name).write_text(text)
    return a, b


def test_check_gboot_identical_dirs_pass(reb, tmp_path):
    a, b = _boot_dirs(tmp_path)
    out = reb.check_gboot(a, b)
    assert out["n_files_compared"] == 2
    assert out["canonical_dir"] == str(a)
    assert out["replicate_dir"] == str(b)
    assert out["files_sha256"]["split_daily.csv"] == _sha(a / "split_daily.csv")


def test_check_gboot_perturbed_file_raises(reb, tmp_path):
    a, b = _boot_dirs(tmp_path)
    (b / "deltas.csv").write_text("d,0.2\n")
    with pytest.raises(reb.BenchmarkConstructionError, match="deltas.csv differs"):
        reb.check_gboot(a, b)


def test_check_gboot_missing_replicate_file_raises(reb, tmp_path):
    a, b = _boot_dirs(tmp_path)
    (b / "split_daily.csv").unlink()
    with pytest.raises(reb.BenchmarkConstructionError, match="missing from replicate"):
        reb.check_gboot(a, b)


def test_check_gboot_empty_canonical_raises(reb, tmp_path):
    a = tmp_path / "empty"
    a.mkdir()
    with pytest.raises(reb.BenchmarkConstructionError, match="no CSV files"):
        reb.check_gboot(a, tmp_path)


# ---------------------------------------------------------- compare_to_pinned


@pytest.fixture
def pinned_state(tmp_path):
    """A pinned metadata dict plus an out_dir/temporal dir consistent with it."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    artifact = out_dir / "e2_primary_et_performance.csv"
    artifact.write_text("site,nse\nA,0.7\n")
    tdir = tmp_path / "temporal"
    tdir.mkdir()
    tfile = tdir / "overpass_split_paired_deltas.csv"
    tfile.write_text("cohort,delta\ncommon_split,0.004\n")
    pinned = {
        "headline": {
            "daily_swim": {"nse": 0.699145, "n_sites": 45},
            "daily_ensemble": {"nse": 0.716052, "n_sites": 45},
        },
        "scientific_configuration": {"window_days": 32, "eto_source": "openet_eto.csv"},
        "support_reconciliation": {
            "n_rows": 315,
            "series": ["eemetric", "ensemble"],
            "per_series_scored_days": {
                "eemetric": {"capture": 100, "interpolated": 900},
                "ensemble": {"capture": 110, "interpolated": 890},
            },
        },
        "source_audit": {"daily_files_validated": 151, "monthly_files_validated": 125},
        "source_data": {"input_sha256": {"/data/may/daily/A.csv": "abc123"}},
        "frozen_artifacts": {artifact.name: _sha(artifact)},
        "temporal_artifacts": {
            "canonical_dir": str(tdir),
            "files_sha256": {tfile.name: _sha(tfile)},
        },
    }
    return pinned, out_dir, artifact, tfile


def test_compare_clean_pass(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    assert reb.compare_to_pinned(pinned, copy.deepcopy(pinned), out_dir) == []


def test_compare_headline_drift_fails(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["headline"]["daily_swim"]["nse"] = 0.75
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("headline.daily_swim.nse" in f for f in failures)


def test_compare_missing_headline_block_fails(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    del new["headline"]["daily_ensemble"]
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("daily_ensemble: missing" in f for f in failures)


def test_compare_scalar_block_drift_fails(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["source_audit"]["daily_files_validated"] = 59
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("source_audit.daily_files_validated" in f for f in failures)


def test_compare_per_series_counts_fail(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["support_reconciliation"]["per_series_scored_days"]["eemetric"]["capture"] = 99
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("per_series_scored_days" in f for f in failures)


def test_compare_series_list_fail(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["support_reconciliation"]["series"] = ["eemetric"]
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("support_reconciliation.series" in f for f in failures)


def test_compare_input_hash_change_fails(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["source_data"]["input_sha256"]["/data/may/daily/A.csv"] = "def456"
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("input_sha256" in f and "changed or missing" in f for f in failures)


def test_compare_unpinned_extra_input_fails(reb, pinned_state):
    pinned, out_dir, _, _ = pinned_state
    new = copy.deepcopy(pinned)
    new["source_data"]["input_sha256"]["/data/may/daily/B.csv"] = "fff999"
    failures = reb.compare_to_pinned(pinned, new, out_dir)
    assert any("read now but not pinned" in f for f in failures)


def test_compare_modified_output_fails(reb, pinned_state):
    pinned, out_dir, artifact, _ = pinned_state
    artifact.write_text("site,nse\nA,0.9\n")
    failures = reb.compare_to_pinned(pinned, copy.deepcopy(pinned), out_dir)
    assert any("hash differs from pinned" in f for f in failures)


def test_compare_missing_output_fails(reb, pinned_state):
    pinned, out_dir, artifact, _ = pinned_state
    artifact.unlink()
    failures = reb.compare_to_pinned(pinned, copy.deepcopy(pinned), out_dir)
    assert any(f"frozen_artifacts: {artifact.name} missing" in f for f in failures)


def test_compare_temporal_artifact_drift_fails(reb, pinned_state):
    pinned, out_dir, _, tfile = pinned_state
    tfile.write_text("cohort,delta\ncommon_split,0.999\n")
    failures = reb.compare_to_pinned(pinned, copy.deepcopy(pinned), out_dir)
    assert any("temporal_artifacts" in f and "hash differs" in f for f in failures)
