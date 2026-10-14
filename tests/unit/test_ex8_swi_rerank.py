"""Pure unit tests for the Example 8 posterior SWI reranker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "8_Soil_Moisture" / "posterior_swi_rerank.py"
)


@pytest.fixture(scope="module")
def rerank():
    spec = importlib.util.spec_from_file_location("ex8_posterior_swi_rerank", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site_rows(realization, train_r, holdout_r, et_mae, et_bias=0.01):
    return pd.DataFrame(
        {
            "realization": [realization, realization],
            "site_id": ["A", "B"],
            "scheduler_active": [True, False],
            "swi_train_anom_r": [train_r, train_r - 0.1],
            "swi_train_anom_rmse": [0.02, 0.03],
            "swi_train_std_ratio": [0.9, 0.8],
            "swi_holdout_anom_r": [holdout_r, holdout_r - 0.1],
            "swi_holdout_anom_rmse": [0.03, 0.04],
            "swi_holdout_std_ratio": [0.85, 0.75],
            "et_n": [100, 100],
            "et_mae": [et_mae, et_mae],
            "et_abs_bias": [et_bias, et_bias],
        }
    )


def test_parse_row_params_handles_field_and_parameter_underscores(rerank):
    row = pd.Series(
        {
            "pname:p_min_irr_days_field_one_:0_ptype:cn_usecol:1_pstyle:m": 12.0,
            "pname:p_aw_b_:0_ptype:cn_usecol:1_pstyle:m": 220.0,
        }
    )
    got = rerank.parse_row_params(row, ["Field_One", "B"])
    assert got == {"Field_One": {"min_irr_days": 12.0}, "B": {"aw": 220.0}}


def test_summarize_realization_keeps_scheduler_active_cohort(rerank):
    summary = rerank.summarize_realization(_site_rows("7", 0.7, 0.5, 0.15))
    assert summary["n_scheduler_active"] == 1
    assert summary["swi_train_active_anom_r"] == pytest.approx(0.7)
    assert summary["et_active_mae"] == pytest.approx(0.15)


def test_et_gates_reject_phi_median_bias_and_site_regressions(rerank):
    frames = [
        _site_rows(rerank.BASELINE, 0.4, 0.4, 0.15, et_bias=0.01),
        _site_rows("good", 0.6, 0.5, 0.151, et_bias=0.009),
        _site_rows("phi_bad", 0.7, 0.6, 0.15, et_bias=0.009),
        _site_rows("median_bad", 0.7, 0.6, 0.154, et_bias=0.009),
        _site_rows("bias_bad", 0.7, 0.6, 0.15, et_bias=0.011),
        _site_rows("site_bad", 0.7, 0.6, 0.15, et_bias=0.009),
    ]
    frames[-1].loc[frames[-1]["site_id"] == "B", "et_mae"] = 0.171
    sites = pd.concat(frames, ignore_index=True)
    summaries = pd.DataFrame(
        [rerank.summarize_realization(group) for _, group in sites.groupby("realization")]
    )
    phi = {
        rerank.BASELINE: 100.0,
        "good": 101.0,
        "phi_bad": 103.0,
        "median_bad": 101.0,
        "bias_bad": 101.0,
        "site_bad": 101.0,
    }
    summaries["phi"] = summaries["realization"].map(phi)
    got = rerank.apply_et_gates(summaries, sites, phi_reference=100.0).set_index("realization")
    assert bool(got.loc["good", "et_eligible"])
    for label in ["phi_bad", "median_bad", "bias_bad", "site_bad"]:
        assert not bool(got.loc[label, "et_eligible"])


def test_selector_uses_training_not_holdout(rerank):
    summaries = pd.DataFrame(
        {
            "realization": [rerank.BASELINE, "train_winner", "holdout_winner"],
            "et_eligible": [True, True, True],
            "swi_train_active_anom_r": [0.4, 0.8, 0.7],
            "swi_train_active_rmse": [0.04, 0.03, 0.02],
            "swi_train_active_std_ratio": [0.6, 0.9, 1.0],
            "swi_holdout_active_anom_r": [0.4, 0.1, 0.95],
        }
    )
    assert rerank.select_training_candidate(summaries)["realization"] == "train_winner"


def test_pareto_front_excludes_dominated_member(rerank):
    summaries = pd.DataFrame(
        {
            "realization": ["a", "b", "c"],
            "et_eligible": [True, True, True],
            "swi_train_active_anom_r": [0.6, 0.5, 0.7],
            "et_active_mae": [0.15, 0.16, 0.17],
        }
    )
    got = rerank.mark_pareto_front(summaries).set_index("realization")
    assert bool(got.loc["a", "pareto_front"])
    assert not bool(got.loc["b", "pareto_front"])
    assert bool(got.loc["c", "pareto_front"])
