"""Unit tests for the E2 closure-pool gates in scripts/figures/build_figure_data.py.

The builder restricts Fig. 1 and Fig. 5 to the paper's 47 closure-corrected
E2 towers (frozen 2026-09-21). These tests exercise the gate helpers on
synthetic frames so a regression in the pool logic fails here rather than in
a full figure build.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts" / "figures" / "build_figure_data.py"


@pytest.fixture(scope="module")
def bfd():
    spec = importlib.util.spec_from_file_location("build_figure_data", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


N_POOL = 47
N_ALL = 63
N_OVERLAP_POOL = 10
N_CONFIGURED = 66
CFGS = ("e3_uncal", "ex5_transfer_strat", "e3_cal", "ls_ensemble")


def _site_ids(n, prefix="S"):
    return [f"{prefix}-{i:03d}" for i in range(n)]


def _pool_frame(ids):
    countries = ["US"] * 29 + ["AU"] * 5 + ["IT"] * 4 + ["DE"] * 3 + ["FR"] * 3 + ["BE", "CA", "CR"]
    continents = {
        "US": "North America",
        "CA": "North America",
        "CR": "North America",
        "AU": "Oceania",
        "IT": "Europe",
        "DE": "Europe",
        "FR": "Europe",
        "BE": "Europe",
    }
    assert len(countries) == N_POOL
    return pd.DataFrame(
        {
            "fid": ids,
            "closure_tier": "closure_corrected",
            "country": countries,
            "continent": [continents[c] for c in countries],
        }
    )


def _persite_frame(ids, seed=0):
    rng = np.random.default_rng(seed)
    data = {"site_id": ids}
    for cfg in CFGS:
        for col in ("kge", "r2", "rmse", "bias", "mae"):
            data[f"{cfg}_{col}"] = rng.normal(size=len(ids))
    return pd.DataFrame(data).set_index("site_id")


def _summary_from(ps):
    rows = []
    for label, cfg in {
        "E3 uncalibrated/default": "e3_uncal",
        "Ex5 stratified transfer": "ex5_transfer_strat",
        "E3 calibrated": "e3_cal",
        "LS ensemble": "ls_ensemble",
    }.items():
        rows.append(
            {
                "config": label,
                "basis": "daily",
                "n_sites": N_POOL,
                "kge_med": ps[f"{cfg}_kge"].median(),
                "rmse_med": ps[f"{cfg}_rmse"].median(),
                "bias_med": ps[f"{cfg}_bias"].median(),
            }
        )
        if cfg != "e3_uncal":
            rows.append(
                {
                    "config": label,
                    "basis": "monthly",
                    "n_sites": 39,
                    "kge_med": 0.7,
                    "rmse_med": 19.0,
                    "bias_med": 0.0,
                }
            )
    rows.append(
        {
            "config": "something not reported",
            "basis": "daily",
            "n_sites": 63,
            "kge_med": 0.0,
            "rmse_med": 0.0,
            "bias_med": 0.0,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def pool_ids():
    return _site_ids(N_POOL)


@pytest.fixture
def all_ids(pool_ids):
    return pool_ids + _site_ids(N_ALL - N_POOL, prefix="RAW")


# ---------------------------------------------------------------------------
# gate_closure_pool / restrict_to_pool (Fig. 5 E2)
# ---------------------------------------------------------------------------


def test_gate_closure_pool_accepts_frozen_pool(bfd, pool_ids):
    got = bfd.gate_closure_pool(_pool_frame(pool_ids), "t")
    assert got == set(pool_ids)


def test_gate_closure_pool_rejects_wrong_count(bfd, pool_ids):
    with pytest.raises(bfd.BuildError):
        bfd.gate_closure_pool(_pool_frame(pool_ids).iloc[:-1], "t")


def test_gate_closure_pool_rejects_duplicate_site(bfd, pool_ids):
    pool = _pool_frame(pool_ids)
    pool.loc[1, "fid"] = pool.loc[0, "fid"]
    with pytest.raises(bfd.BuildError):
        bfd.gate_closure_pool(pool, "t")


def test_gate_closure_pool_rejects_raw_et_tier(bfd, pool_ids):
    pool = _pool_frame(pool_ids)
    pool.loc[3, "closure_tier"] = "raw"
    with pytest.raises(bfd.BuildError, match="non-closure-corrected"):
        bfd.gate_closure_pool(pool, "t")


def test_restrict_to_pool_keeps_only_pool_sites(bfd, pool_ids, all_ids):
    ps_all = _persite_frame(all_ids)
    ps = bfd.restrict_to_pool(ps_all, set(pool_ids), N_POOL, "t")
    assert len(ps) == N_POOL
    assert set(ps.index) == set(pool_ids)
    # the source table is untouched
    assert len(ps_all) == N_ALL


def test_restrict_to_pool_rejects_missing_pool_site(bfd, pool_ids, all_ids):
    ps_all = _persite_frame(all_ids).drop(index=pool_ids[0])
    with pytest.raises(bfd.BuildError, match="absent"):
        bfd.restrict_to_pool(ps_all, set(pool_ids), N_POOL, "t")


def test_restrict_to_pool_rejects_wrong_expected_count(bfd, pool_ids, all_ids):
    with pytest.raises(bfd.BuildError):
        bfd.restrict_to_pool(_persite_frame(all_ids), set(pool_ids), N_ALL, "t")


# ---------------------------------------------------------------------------
# gate_pool_summary / check_pool_medians (Fig. 5 E2)
# ---------------------------------------------------------------------------


def test_gate_pool_summary_keeps_seven_reported_rows(bfd, pool_ids):
    ps = _persite_frame(pool_ids)
    summ = bfd.gate_pool_summary(_summary_from(ps), "t")
    assert len(summ) == 7
    assert set(summ["legacy_config"]) == set(CFGS)
    assert (summ.loc[summ["basis"] == "daily", "n_sites"] == N_POOL).all()
    assert (summ.loc[summ["basis"] == "monthly", "n_sites"] == 39).all()


def test_gate_pool_summary_rejects_daily_rows_off_pool(bfd, pool_ids):
    summ = _summary_from(_persite_frame(pool_ids))
    summ.loc[(summ["basis"] == "daily") & (summ["config"] == "E3 calibrated"), "n_sites"] = 63
    with pytest.raises(bfd.BuildError, match="47-site pool"):
        bfd.gate_pool_summary(summ, "t")


def test_gate_pool_summary_rejects_monthly_rows_off_finite_pool(bfd, pool_ids):
    summ = _summary_from(_persite_frame(pool_ids))
    summ.loc[summ["basis"] == "monthly", "n_sites"] = 43
    with pytest.raises(bfd.BuildError, match="39 finite-metric"):
        bfd.gate_pool_summary(summ, "t")


def test_check_pool_medians_passes_when_summary_matches(bfd, pool_ids):
    ps = _persite_frame(pool_ids)
    summ = bfd.gate_pool_summary(_summary_from(ps), "t")
    for cfg in CFGS:
        bfd.check_pool_medians(ps, summ, cfg, "t")


def test_check_pool_medians_rejects_stale_summary(bfd, pool_ids):
    ps = _persite_frame(pool_ids)
    summ = bfd.gate_pool_summary(_summary_from(ps), "t")
    # a summary computed on a different per-site table (e.g. a superseded run)
    stale = _persite_frame(pool_ids, seed=1)
    with pytest.raises(bfd.BuildError, match="pool median"):
        bfd.check_pool_medians(stale, summ, "e3_cal", "t")


def test_check_pool_medians_tolerance(bfd, pool_ids):
    ps = _persite_frame(pool_ids)
    summ = bfd.gate_pool_summary(_summary_from(ps), "t")
    summ.loc[(summ["legacy_config"] == "e3_cal") & (summ["basis"] == "daily"), "kge_med"] += 5e-7
    bfd.check_pool_medians(ps, summ, "e3_cal", "t")
    summ.loc[(summ["legacy_config"] == "e3_cal") & (summ["basis"] == "daily"), "kge_med"] += 5e-6
    with pytest.raises(bfd.BuildError):
        bfd.check_pool_medians(ps, summ, "e3_cal", "t")


# ---------------------------------------------------------------------------
# flag_evaluation_pool (Fig. 1)
# ---------------------------------------------------------------------------


def _scope_frame(pool_ids):
    ids = pool_ids + _site_ids(N_CONFIGURED - N_POOL, prefix="CFG")
    in_e1 = [False] * len(ids)
    for i in range(N_OVERLAP_POOL):
        in_e1[i] = True  # overlap sites inside the pool
    in_e1[N_POOL] = in_e1[N_POOL + 1] = in_e1[N_POOL + 2] = True  # 3 overlap sites outside
    return pd.DataFrame({"site_id": ids, "in_e1": in_e1})


def test_flag_evaluation_pool_flags_pool_and_counts(bfd, pool_ids):
    e2 = _scope_frame(pool_ids)
    flagged, n_countries, n_continents = bfd.flag_evaluation_pool(e2, _pool_frame(pool_ids), "t")
    assert flagged["in_evaluation_pool"].sum() == N_POOL
    assert set(flagged.loc[flagged["in_evaluation_pool"], "site_id"]) == set(pool_ids)
    assert (flagged["in_evaluation_pool"] & flagged["in_e1"]).sum() == N_OVERLAP_POOL
    assert flagged["in_e1"].sum() == 13
    assert (n_countries, n_continents) == (8, 3)
    assert "in_evaluation_pool" not in e2.columns  # input frame untouched


def test_flag_evaluation_pool_rejects_pool_site_outside_scope(bfd, pool_ids):
    e2 = _scope_frame(pool_ids)
    e2.loc[0, "site_id"] = "NOT-IN-POOL"
    with pytest.raises(bfd.BuildError, match="not in the configured E2 scope"):
        bfd.flag_evaluation_pool(e2, _pool_frame(pool_ids), "t")


def test_flag_evaluation_pool_rejects_wrong_overlap(bfd, pool_ids):
    e2 = _scope_frame(pool_ids)
    e2.loc[N_OVERLAP_POOL, "in_e1"] = True  # 11 overlap sites inside the pool
    with pytest.raises(bfd.BuildError, match="overlap within the E2 pool"):
        bfd.flag_evaluation_pool(e2, _pool_frame(pool_ids), "t")


def test_flag_evaluation_pool_rejects_country_count_drift(bfd, pool_ids):
    pool = _pool_frame(pool_ids)
    pool.loc[0, "country"] = "MX"
    with pytest.raises(bfd.BuildError, match="pool countries"):
        bfd.flag_evaluation_pool(_scope_frame(pool_ids), pool, "t")


def test_flag_evaluation_pool_rejects_continent_count_drift(bfd, pool_ids):
    pool = _pool_frame(pool_ids)
    pool.loc[0, "continent"] = "South America"
    with pytest.raises(bfd.BuildError, match="pool continents"):
        bfd.flag_evaluation_pool(_scope_frame(pool_ids), pool, "t")


def test_flag_evaluation_pool_rejects_raw_tier_in_pool(bfd, pool_ids):
    pool = _pool_frame(pool_ids)
    pool.loc[5, "closure_tier"] = "raw"
    with pytest.raises(bfd.BuildError, match="non-closure-corrected"):
        bfd.flag_evaluation_pool(_scope_frame(pool_ids), pool, "t")


def test_expected_pool_constants(bfd):
    """The gate constants are the paper's frozen pool numbers (decision 2026-09-08)."""
    e = bfd.EXPECTED
    assert e["E2_pool_daily"] == 47
    assert e["E2_pool_monthly_support"] == 43
    assert e["E2_pool_monthly_finite"] == 39
    assert e["E2_pool_conus"] + e["E2_pool_ex_conus"] == 47
    assert (e["E2_pool_countries"], e["E2_pool_continents"]) == (8, 3)
    assert e["E1_E2_overlap_pool"] <= e["E1_E2_overlap"]
