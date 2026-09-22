"""``6_Flux_International/promote_final.py`` copies the E2 results into the frozen package."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EX6 = Path(__file__).resolve().parents[2] / "examples" / "6_Flux_International"


@pytest.fixture(scope="module")
def mod():
    if str(EX6) not in sys.path:
        sys.path.insert(0, str(EX6))
    spec = importlib.util.spec_from_file_location("e2_promote_final", EX6 / "promote_final.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADLINE = (
    "basis,tier,model,n_rows,n_sites,r2_median,r2_mean,kge_median,kge_mean,"
    "rmse_median,rmse_mean,bias_median,bias_mean\n"
    "daily,closure_corrected,swim,47,47,0.6,0.2,0.7,0.6,1.0,1.1,-0.03,-0.1\n"
    "daily,closure_corrected,rs,47,47,0.66,0.45,0.73,0.64,0.96,1.06,-0.1,-0.14\n"
    "daily,raw,swim,16,16,0.5,0.4,0.7,0.6,1.2,1.3,0.1,0.1\n"
    "monthly,closure_corrected,swim,43,39,0.74,-0.26,0.73,0.61,18.7,23.7,-0.74,-3.0\n"
)
POOL_META = {
    "created_utc": "2026-09-21T23:00:41+00:00",
    "git_sha": "0b8b547",
    "decision": "EBR towers only",
    "pool_definition": {"column": "flux_et_col", "value": "ET_corr"},
    "n_daily_sites_all": 63,
    "n_daily_sites_pool": 47,
    "n_daily_sites_raw_excluded": 16,
    "n_monthly_rows_pool": 43,
    "n_monthly_finite_pool": 39,
    "n_paired_days_pool": 59343,
    "n_paired_days_all": 80501,
    "irrigation_class_pool": {"rainfed": 36, "irrigated": 11},
    "configured_pool_note": "note",
    "bootstrap": {"replicates": 10, "seed": 1},
    "uncalibrated_baseline": {"canonical_file": "uncalibrated_baseline_summary.csv"},
    "problems": [],
}
GATE = {
    "arm_a": "grassbasis",
    "arm_b": "fao56",
    "a_config": "/x/a.toml",
    "n_sites": 37,
    "n_daily": 49289,
    "n_monthly": 1514,
    "wins_a": 5,
    "gate_rule": "arm A wins >= 4 of 6 pooled metrics",
    "passed": True,
}


def _results_root(mod, tmp_path):
    results = tmp_path / "results"
    run_dir = results / mod.ex6_paths.CANONICAL_RUN
    closure = run_dir / "archive" / "6_evaluation" / "closure_pool"
    closure.mkdir(parents=True)
    (closure / "headline_aggregates.csv").write_text(HEADLINE)
    (closure / "closure_pool_metadata.json").write_text(json.dumps(POOL_META))
    (closure / "uncalibrated_baseline_summary.csv").write_text(
        "series,n_sites,kge_median\ncal,47,0.7\nuncal,47,0.54\n"
    )
    (closure / "transfer_refresh_summary.csv").write_text(
        "application,config,basis,n_sites,kge_med\nrefresh,E3 calibrated,daily,47,0.7\n"
    )
    (run_dir / "archive" / "1_provenance").mkdir()
    (run_dir / "archive" / "1_provenance" / "git_sha.txt").write_text("0b8b547\n")
    (run_dir / "transfer_refresh").mkdir()
    for name in mod.MAPPING_FILES:
        (run_dir / "transfer_refresh" / name).write_text("{}\n")
    transfer = results / mod.TRANSFER_RUN
    transfer.mkdir()
    for name in mod.TRANSFER_FILES:
        (transfer / name).write_text(f"{name}\n")
    (transfer / "AR-CCa.csv").write_text("per-site output, not promoted\n")
    for suffix in mod.E0_ARMS.values():
        (results / f"{mod.ex6_paths.CANONICAL_RUN}_{suffix}").mkdir()
    for pair in mod.E0_PAIRS:
        for tag in mod.E0_TAGS:
            d = results / "e0_disjoint" / pair / tag
            d.mkdir(parents=True)
            (d / "pooled_gate.json").write_text(json.dumps({**GATE, "arm_b": pair}))
            (d / "pooled_per_site.csv").write_text("site,kge\nA,0.5\n")
    return results, run_dir, transfer, results / "e0_disjoint"


def _final_dir(tmp_path):
    final = tmp_path / "final" / "e2_closure_pool"
    final.mkdir(parents=True)
    for name in (
        "e2_evidence_metadata.json",
        "e2_run22_transfer_vector.json",
        "e2_run22_transfer_vectors_by_irrigation.json",
    ):
        (final.parent / name).write_text("{}\n")
    return final


def test_collect_sources_maps_every_promoted_file(mod, tmp_path):
    results, run_dir, transfer, e0 = _results_root(mod, tmp_path)
    sources = mod.collect_sources(run_dir, transfer, e0)
    assert len(sources) == 4 + len(mod.TRANSFER_FILES) + len(mod.MAPPING_FILES) + 3 * 2 * 2
    assert sources["transfer/run_metadata.json"] == transfer / "run_metadata.json"
    assert sources["transfer/" + mod.MAPPING_FILES[0]].parent == run_dir / "transfer_refresh"
    assert "transfer/AR-CCa.csv" not in sources
    assert all(k.startswith(("closure_pool/", "transfer/", "e0_disjoint/")) for k in sources)


def test_collect_sources_fails_on_a_missing_transfer_output(mod, tmp_path):
    results, run_dir, transfer, e0 = _results_root(mod, tmp_path)
    (transfer / "transfer_winrates.csv").unlink()
    with pytest.raises(FileNotFoundError, match="transfer_winrates.csv"):
        mod.collect_sources(run_dir, transfer, e0)


def test_compare_and_manifest_round_trip(mod, tmp_path):
    results, run_dir, transfer, e0 = _results_root(mod, tmp_path)
    final = _final_dir(tmp_path)
    sources = mod.collect_sources(run_dir, transfer, e0)
    assert {s for _, s, _ in mod.compare(sources, final)} == {"missing"}

    prior = {
        "reason": "why",
        "e0_gate_note": "note",
        "source_runs": {"canonical": {"supersedes": "/old"}, "transfer": {"supersedes": "/old_t"}},
        "uncalibrated_baseline": {"note": "mask note"},
    }
    manifest = mod.build_manifest(sources, prior, results, run_dir, transfer, e0, final)
    assert manifest["schema_version"] == mod.SCHEMA
    assert manifest["reason"] == "why" and manifest["e0_gate_note"] == "note"
    assert manifest["source_runs"]["canonical"] == {
        "path": str(run_dir),
        "git_sha": "0b8b547",
        "supersedes": "/old",
    }
    assert manifest["source_runs"]["transfer"] == {"path": str(transfer), "supersedes": "/old_t"}
    assert "git_sha" not in manifest["source_runs"]["e0_disjoint"]
    assert manifest["pool"]["n_daily_sites_pool"] == 47
    assert "n_paired_days_all" not in manifest["pool"]
    assert manifest["uncalibrated_baseline"]["note"] == "mask note"
    assert manifest["uncalibrated_baseline"]["canonical_row"][1]["series"] == "uncal"
    head = manifest["headline_closure_corrected"]
    assert set(head) == {"daily_closure_corrected", "monthly_closure_corrected"}
    assert head["daily_closure_corrected"]["rs"]["kge_median"] == 0.73
    assert "raw" not in json.dumps(head)
    gates = manifest["e0_pooled_gates"]
    assert set(gates) == {f"{p}/{t}" for p in mod.E0_PAIRS for t in mod.E0_TAGS}
    assert gates["grassbasis_vs_fao56/disjoint37"]["wins_a"] == 5
    assert "a_config" not in gates["grassbasis_vs_fao56/disjoint37"]
    assert set(manifest["files"]) == set(sources)
    entry = manifest["files"]["transfer/run_metadata.json"]
    assert entry["source"] == str(transfer / "run_metadata.json")
    assert entry["bytes"] == len("run_metadata.json\n")

    for name, src in sources.items():
        (final / name).parent.mkdir(parents=True, exist_ok=True)
        (final / name).write_bytes(src.read_bytes())
    assert {s for _, s, _ in mod.compare(sources, final)} == {"match"}
    assert mod.manifest_diff(manifest, manifest) == []

    (final / "closure_pool" / "headline_aggregates.csv").write_text("changed\n")
    statuses = {n: s for n, s, _ in mod.compare(sources, final)}
    assert statuses["closure_pool/headline_aggregates.csv"] == "differs"
    other = dict(manifest, promoted_at="later", pool={**manifest["pool"], "n_daily_sites_pool": 1})
    assert mod.manifest_diff(other, manifest) == ["pool"]
