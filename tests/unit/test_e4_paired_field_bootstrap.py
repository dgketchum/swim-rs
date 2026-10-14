"""Tests for the Example 7 paired field-clustered bootstrap."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "7_Applied_Water"
    / "paired_field_bootstrap.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("e4_paired_field_bootstrap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bootstrap_module():
    return _load_module()


def _write_inputs(tmp_path, transfer_offset=0.0):
    rows = []
    for basin_index, basin in enumerate(("SLV", "ESPA")):
        for field_index in range(3):
            site_id = f"{basin}_{field_index}"
            acres = 100.0 + 10.0 * field_index
            for year_offset, year in enumerate(range(2018, 2023)):
                observed = 400.0 + 35.0 * year_offset + 25.0 * field_index + 60.0 * basin_index
                rows.append(
                    {
                        "site_id": site_id,
                        "year": year,
                        "metered_depth_mm": observed,
                        "metered_volume_af": observed / 304.8 * acres,
                        "acres": acres,
                        "method": "metered",
                        "source": basin,
                        "sim_applied_mm": observed * 0.95,
                        "sim_et_mm": observed * 0.8,
                        "basin": basin,
                        "crop": "TEST",
                    }
                )
    local = pd.DataFrame(rows)
    transfer = local.copy()
    transfer["sim_applied_mm"] += transfer_offset
    local_path = tmp_path / "local.csv"
    transfer_path = tmp_path / "transfer.csv"
    local.to_csv(local_path, index=False)
    transfer.to_csv(transfer_path, index=False)
    return local_path, transfer_path


def test_rejects_nonidentical_field_year_keys(tmp_path, bootstrap_module):
    local_path, transfer_path = _write_inputs(tmp_path)
    transfer = pd.read_csv(transfer_path).iloc[:-1]
    transfer.to_csv(transfer_path, index=False)

    with pytest.raises(ValueError, match="field-year keys differ"):
        bootstrap_module.load_paired_inputs(local_path, transfer_path)


def test_identical_paths_are_deterministic_and_zero(tmp_path, bootstrap_module):
    local_path, transfer_path = _write_inputs(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    summary_a, replicates_a, metadata_a = bootstrap_module.run_analysis(
        local_path,
        transfer_path,
        out_a,
        bootstrap_reps=100,
        seed=42,
    )
    summary_b, replicates_b, metadata_b = bootstrap_module.run_analysis(
        local_path,
        transfer_path,
        out_b,
        bootstrap_reps=100,
        seed=42,
    )

    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(replicates_a, replicates_b)
    assert metadata_a == metadata_b
    finite_deltas = replicates_a.drop(columns=["scope", "replicate"]).to_numpy(float)
    assert np.nanmax(np.abs(finite_deltas)) == 0.0
    assert set(metadata_a["basin_field_counts"].values()) == {3}


def test_sampling_preserves_each_stratum_size(bootstrap_module):
    strata = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]
    sampled = bootstrap_module._sample_field_indices(strata, np.random.default_rng(5))

    assert len(sampled) == 7
    assert np.isin(sampled[:3], strata[0]).all()
    assert np.isin(sampled[3:], strata[1]).all()


def test_negative_control_mismatch_fails(bootstrap_module):
    local = {
        "n_control_fields": 10,
        "n_control_field_years": 250,
        "sim_applied_mm_max": 0.0,
        "frac_years_gt_10mm": 0.0,
    }
    transfer = dict(local, sim_applied_mm_max=1.0)

    with pytest.raises(ValueError, match="nonzero simulated irrigation"):
        bootstrap_module._validate_control_summaries(local, transfer)


def test_point_estimate_reconciliation_passes_and_rejects_mismatch(tmp_path, bootstrap_module):
    rows = []
    archived = {}
    for metric, (group, key, decimals) in bootstrap_module.POINT_RECONCILIATION.items():
        value = 1.234
        rows.append({"scope": "all", "metric": metric, "local_value": value})
        archived.setdefault(group, {})[key] = round(value, decimals)

    summary = pd.DataFrame(rows)
    stats_path = tmp_path / "field_accuracy_stats.json"
    stats_path.write_text(json.dumps(archived))
    reconciliation = bootstrap_module._reconcile_point_estimates(summary, stats_path, "local_value")

    assert reconciliation["status"] == "passed"
    assert reconciliation["n_checks"] == len(bootstrap_module.POINT_RECONCILIATION)

    archived["per_field_volume_total"]["r"] = 99.0
    stats_path.write_text(json.dumps(archived))
    with pytest.raises(ValueError, match="Point-estimate reconciliation failed"):
        bootstrap_module._reconcile_point_estimates(summary, stats_path, "local_value")
