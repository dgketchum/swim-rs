"""Consumer tests for the E1 temporal decomposition script.

``overpass_decomposition.py`` is a strict downstream consumer of the
evaluator-owned paired-record bundle (``evaluation_paired_daily_records.csv``,
schema ``e1_openet_paired_daily/v1``) written by ``evaluate.py``. These tests
verify the consumer contract:

- the CLI consumes an evaluator output directory and rejects the removed
  raw-input arguments (no reconstruction path exists);
- the parent-artifact identity gate re-hashes every artifact in
  ``output_hashes``, verifies the record contract, and recomputes the all-days
  grouped points within 1e-12;
- a DIY bundle, a hash-mismatched bundle, a stale/corrupted metadata sidecar,
  an unsupported record schema, and a missing parent artifact all hard-fail
  before any temporal output is written;
- legacy site products appear only with ``--legacy-site-products``;
- outputs are deterministic (byte-identical across reruns) and the metadata
  sidecar, written last, hash-covers every CSV.

The old reconstruction/classification behavior is covered by
``tests/unit/test_benchmark_reconstruction.py`` — none of it lives in this
script any more.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"

OUT_FILES = [
    "evaluation_temporal_grouped_metrics.csv",
    "evaluation_temporal_grouped_contrasts.csv",
    "evaluation_temporal_interactions.csv",
    "evaluation_temporal_site_eligibility.csv",
    "evaluation_temporal_metadata.json",
]
LEGACY_FILES = [
    "overpass_split_metrics.csv",
    "overpass_split_summary.csv",
    "overpass_split_paired_deltas.csv",
    "overpass_date_audit.csv",
    "e2_temporal_support_contrast.csv",
    "e2_temporal_support_contrast_persite.csv",
    "overpass_split_metadata.json",
]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ev():
    return _load("e2_evaluate_for_od", "evaluate.py")


@pytest.fixture(scope="module")
def od():
    return _load("e1_overpass_decomposition", "overpass_decomposition.py")


def _support(n):
    """Captures every 8th day (>= 10 per site for n >= 80), flat-fill tail."""
    out = []
    for i in range(n):
        if i % 8 == 0:
            out.append("capture")
        elif i >= n - 3:
            out.append("flat_fill")
        else:
            out.append("interpolated")
    return tuple(out)


def _record(ev, fid, n, seed, bias=0.0, scale_mod=1.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    obs = np.abs(rng.normal(3.5, 1.2, size=n)) + 0.5
    swim = obs * scale_mod + bias + rng.normal(0.0, 0.4, size=n)
    openet = obs + rng.normal(0.0, 0.4, size=n) - 0.3
    return ev.PairedSiteSeries(
        fid=fid, index=idx, observed=obs, swim=swim, openet=openet, support_class=_support(n)
    )


def _cohort(ev):
    # every site has >= 10 paired dates in BOTH temporal classes
    return (
        _record(ev, "US-Aaa", 100, seed=21, bias=0.4, scale_mod=1.1),
        _record(ev, "US-Bbb", 160, seed=22, bias=-0.5, scale_mod=0.9),
        _record(ev, "US-Ccc", 300, seed=23, bias=0.05),
    )


def _write_bundle(ev, out_dir, records, scale="daily", openet_source="volk", reps=25):
    metrics, contrasts = ev.grouped_metric_tables(records, scale, reps=reps, seed=42)
    site_metrics = pd.DataFrame(
        {"n": [r.n for r in records]}, index=pd.Index([r.fid for r in records], name="fid")
    )
    # production evaluate.py carries openet_source in via collect_meta
    meta = ev.grouped_metadata(
        records, scale, reps, 42, openet_source, {"openet_source": openet_source}
    )
    bundle = ev.BenchmarkEvaluation(
        site_metrics=site_metrics,
        grouped_metrics=metrics,
        grouped_contrasts=contrasts,
        paired_records=tuple(records),
        site_effect_summary=None,
        metadata=meta,
    )
    return ev.write_grouped_outputs(bundle, str(out_dir), scale, openet_source=openet_source)


@pytest.fixture(scope="module")
def canonical_bundle_dir(ev, tmp_path_factory):
    d = tmp_path_factory.mktemp("evaluator_bundle")
    _write_bundle(ev, d, _cohort(ev))
    return d


def _run_main(od, monkeypatch, bundle_dir, out_dir, extra=()):
    argv = [
        "overpass_decomposition.py",
        "--evaluator-output-dir",
        str(bundle_dir),
        "--output-dir",
        str(out_dir),
        "--bootstrap-reps",
        "25",
        "--seed",
        "42",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    od.main()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_help_describes_evaluator_consumption(od, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["overpass_decomposition.py", "--help"])
    with pytest.raises(SystemExit) as e:
        od.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--evaluator-output-dir" in out
    assert "paired" in out.lower() or "evaluator" in out.lower()
    assert "--legacy-site-products" in out


@pytest.mark.parametrize(
    "removed",
    ["--run-dir", "--openet-daily-dir", "--cohort-csv", "--openet-eto-csv"],
)
def test_removed_raw_input_args_rejected(od, monkeypatch, capsys, tmp_path, removed):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "overpass_decomposition.py",
            "--evaluator-output-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            removed,
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as e:
        od.main()
    assert e.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_no_reconstruction_code_in_consumer(od):
    src = (EXAMPLE_DIR / "overpass_decomposition.py").read_text()
    for banned in (
        "reconstruct_daily_benchmark",
        "reconstruct_benchmark",
        "SwimContainer",
        "ProjectConfig",
        "read_flux",
        "is_overpass",
    ):
        assert banned not in src, banned


# ---------------------------------------------------------------------------
# Canonical bundle: success path
# ---------------------------------------------------------------------------


def test_canonical_bundle_succeeds(od, monkeypatch, canonical_bundle_dir, tmp_path):
    _run_main(od, monkeypatch, canonical_bundle_dir, tmp_path)
    for name in OUT_FILES:
        assert (tmp_path / name).is_file(), name
    for name in LEGACY_FILES:
        assert not (tmp_path / name).exists(), name

    inter = pd.read_csv(tmp_path / "evaluation_temporal_interactions.csv")
    assert len(inter) == 6
    assert set(inter["interaction"]) == {"between_retrieval_minus_retrieval_of_swim_minus_openet"}

    meta = json.loads((tmp_path / "evaluation_temporal_metadata.json").read_text())
    assert meta["record_schema"] == "e1_openet_paired_daily/v1"
    assert meta["cohort"]["n_common_sites"] == 3
    assert meta["parent_bundle"]["grouped_point_identity_max_abs_diff"] <= 1e-12
    assert meta["legacy_site_products"] == {"emitted": False}
    assert meta["estimands"]["primary_conclusion_estimand"] == "cross_model_support_interaction"


def test_metadata_written_last_and_hash_covers_csvs(
    od, monkeypatch, canonical_bundle_dir, tmp_path
):
    _run_main(od, monkeypatch, canonical_bundle_dir, tmp_path)
    meta = json.loads((tmp_path / "evaluation_temporal_metadata.json").read_text())
    hashes = meta["output_hashes"]
    csvs = [n for n in OUT_FILES if n.endswith(".csv")]
    assert sorted(hashes) == sorted(csvs)
    for name, expected in hashes.items():
        assert _sha(tmp_path / name) == expected, name
    # the sidecar is the completion marker, so it must not hash itself
    assert "evaluation_temporal_metadata.json" not in hashes


def test_record_hash_in_parent_gate_report(od, monkeypatch, canonical_bundle_dir, tmp_path):
    _run_main(od, monkeypatch, canonical_bundle_dir, tmp_path)
    meta = json.loads((tmp_path / "evaluation_temporal_metadata.json").read_text())
    rehashed = meta["parent_bundle"]["rehashed_artifacts"]
    assert "evaluation_paired_daily_records.csv" in rehashed
    assert rehashed["evaluation_paired_daily_records.csv"] == _sha(
        canonical_bundle_dir / "evaluation_paired_daily_records.csv"
    )


def test_legacy_products_only_on_request(od, monkeypatch, canonical_bundle_dir, tmp_path):
    _run_main(od, monkeypatch, canonical_bundle_dir, tmp_path, extra=("--legacy-site-products",))
    for name in OUT_FILES + LEGACY_FILES:
        assert (tmp_path / name).is_file(), name
    meta = json.loads((tmp_path / "evaluation_temporal_metadata.json").read_text())
    assert meta["legacy_site_products"]["emitted"] is True
    for name in LEGACY_FILES:
        if name.endswith(".csv"):
            assert meta["output_hashes"][name] == _sha(tmp_path / name)
    legacy_meta = json.loads((tmp_path / "overpass_split_metadata.json").read_text())
    assert "LEGACY/SECONDARY" in legacy_meta["role"]
    assert legacy_meta["subset_labels"] == {
        "overpass": "retrieval",
        "non_overpass": "between_retrieval",
    }
    # the legacy date audit is record-derivable only; calibration-capture
    # auditing is deliberately out of scope for the consumer
    audit = pd.read_csv(tmp_path / "overpass_date_audit.csv")
    assert "n_calibration_captures" not in audit.columns
    assert (
        audit["n_paired_overpass"] + audit["n_paired_non_overpass"] == audit["n_paired_all_days"]
    ).all()


def test_reruns_are_byte_identical(od, monkeypatch, canonical_bundle_dir, tmp_path):
    d1, d2 = tmp_path / "run1", tmp_path / "run2"
    _run_main(od, monkeypatch, canonical_bundle_dir, d1, extra=("--legacy-site-products",))
    _run_main(od, monkeypatch, canonical_bundle_dir, d2, extra=("--legacy-site-products",))
    for name in OUT_FILES + LEGACY_FILES:
        if name.endswith(".csv"):
            assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name


# ---------------------------------------------------------------------------
# Failure modes: every defect is a hard error, never a fallback
# ---------------------------------------------------------------------------


def _assert_fails_before_output(od, monkeypatch, bundle_dir, out_dir, match):
    with pytest.raises(od.ParentBundleError, match=match) as e:
        _run_main(od, monkeypatch, bundle_dir, out_dir)
    assert "rerun evaluate.py" in str(e.value)
    for name in OUT_FILES:
        assert not (out_dir / name).exists(), name


def test_diy_bundle_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "diy_bundle"
    _write_bundle(ev, bundle, _cohort(ev), openet_source="diy")
    _assert_fails_before_output(
        od, monkeypatch, bundle, tmp_path / "out", "missing canonical parent artifact"
    )


def test_monthly_bundle_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "monthly_bundle"
    _write_bundle(ev, bundle, _cohort(ev), scale="monthly")
    _assert_fails_before_output(
        od, monkeypatch, bundle, tmp_path / "out", "missing canonical parent artifact"
    )


def test_missing_parent_artifact_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    (bundle / "evaluation_grouped_daily_contrasts.csv").unlink()
    _assert_fails_before_output(
        od, monkeypatch, bundle, tmp_path / "out", "missing canonical parent artifact"
    )


def test_tampered_record_rejected_by_rehash(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    rec = bundle / "evaluation_paired_daily_records.csv"
    frame = pd.read_csv(rec, float_precision="round_trip")
    frame.loc[0, "swim_et_mm_d"] = frame.loc[0, "swim_et_mm_d"] + 0.001
    frame.to_csv(rec, index=False, float_format="%.17g")
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "hash mismatch")


def test_corrupted_metadata_hash_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    key = "evaluation_paired_daily_records.csv"
    h = meta["output_hashes"][key]
    meta["output_hashes"][key] = ("0" if h[0] != "0" else "1") + h[1:]
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "hash mismatch")


def test_unsupported_schema_version_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["paired_record_contract"]["schema_version"] = "e1_openet_paired_daily/v0"
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "record schema")


def test_contract_sha_disagreement_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    h = meta["paired_record_contract"]["sha256"]
    meta["paired_record_contract"]["sha256"] = ("0" if h[0] != "0" else "1") + h[1:]
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "sha256 disagrees")


def test_stale_contract_counts_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["paired_record_contract"]["n_rows"] += 1
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "n_rows")


def test_grouped_point_mismatch_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    gm_path = bundle / "evaluation_grouped_daily_metrics.csv"
    gm = pd.read_csv(gm_path, float_precision="round_trip")
    gm.loc[0, "estimate"] = gm.loc[0, "estimate"] + 1e-6
    gm.to_csv(gm_path, index=False, float_format="%.17g")
    # keep the rehash gate satisfied so the point-identity gate is what fires
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["output_hashes"]["evaluation_grouped_daily_metrics.csv"] = _sha(gm_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(
        od, monkeypatch, bundle, tmp_path / "out", "recomputed from records differs"
    )


def test_malformed_metadata_scale_rejected(ev, od, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(ev, bundle, _cohort(ev))
    meta_path = bundle / "evaluation_grouped_daily_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["scale"] = "monthly"
    meta_path.write_text(json.dumps(meta, indent=2))
    _assert_fails_before_output(od, monkeypatch, bundle, tmp_path / "out", "not daily")
