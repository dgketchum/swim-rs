"""Unit tests for the E2 closure-corrected pool re-cut (``e2_refooting/phase11_closure_pool_summary.py``).

Covers the pure pieces: closure-tier mapping and pool selection, country/continent from the
site-id prefix (the ``US``/``USA`` normalisation), headline rows with NaN-metric monthly rows,
the paired-median site bootstrap, transfer medians/win rates, the baseline-comparison summary,
irrigation site-year counts, and the Volk-style pooled table reproduced from archived series.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EX6_DIR = REPO_ROOT / "examples" / "6_Flux_International"
E2_DIR = EX6_DIR / "e2_refooting"


# Example-level helper modules share bare names (``evaluate``, ``pooled_metrics``) across
# examples; another test may already have cached the Example 5 ``evaluate``. Evict those names
# while loading so the Example 6 versions resolve, then restore the caller's modules.
_SHARED_NAMES = ("evaluate", "pooled_metrics", "derived_metrics", "phase11_evaluation_summary")


@pytest.fixture(scope="module")
def cp():
    saved = {n: sys.modules.pop(n) for n in _SHARED_NAMES if n in sys.modules}
    for p in (EX6_DIR, E2_DIR):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(
        "phase11_closure_pool_summary", E2_DIR / "phase11_closure_pool_summary.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p not in (str(EX6_DIR), str(E2_DIR))]
        for n in _SHARED_NAMES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return module


def _metrics_frame(fids, seed=0, nan_rows=()):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=pd.Index(fids, name="fid"))
    df["n"] = rng.integers(200, 2000, len(fids))
    for k in ("r2", "r", "alpha", "beta", "kge", "rmse", "mae", "bias"):
        df[f"{k}_swim"] = rng.normal(0.6, 0.1, len(fids))
        df[f"{k}_rs"] = rng.normal(0.65, 0.1, len(fids))
    for fid in nan_rows:
        df.loc[fid, [c for c in df.columns if c != "n"]] = np.nan
    return df


def test_closure_tier_and_pool_selection(cp):
    daily = _metrics_frame(["US-A", "US-B", "DE-C", "AR-D"])
    daily["flux_et_col"] = ["ET_corr", "ET", "ET_corr", "ET"]
    assert cp.select_pool(daily) == ["DE-C", "US-A"]
    assert cp.select_pool(daily, "raw") == ["AR-D", "US-B"]
    daily.loc["US-B", "flux_et_col"] = "LE"
    with pytest.raises(ValueError, match="unmapped"):
        cp.closure_tier(daily)


def test_site_geography_uses_prefix_and_rejects_unknown(cp):
    geo = cp.site_geography(["US-Bi1", "AU-Rgf", "CR-Fsc", "IT-CA2"])
    assert geo.loc["US-Bi1", "country"] == "United States"
    assert geo.loc["AU-Rgf", "continent"] == "Oceania"
    assert geo.loc["CR-Fsc", "continent"] == "North America"
    assert geo.loc["IT-CA2", "continent"] == "Europe"
    with pytest.raises(KeyError):
        cp.site_geography(["ZZ-X"])


def test_headline_rows_count_finite_sites_and_pair_differences(cp):
    frame = _metrics_frame(["a", "b", "c", "d"], nan_rows=["d"])
    rows = {r["model"]: r for r in cp.headline_rows(frame, "monthly", "closure_corrected")}
    assert rows["swim"]["n_rows"] == 4 and rows["swim"]["n_sites"] == 3
    expected = (frame["kge_swim"] - frame["kge_rs"]).median()
    assert rows["swim_minus_rs_paired"]["kge_median"] == pytest.approx(expected)
    assert rows["rs"]["basis"] == "monthly" and rows["rs"]["tier"] == "closure_corrected"


def test_bootstrap_paired_median_brackets_point_estimate_and_ignores_nan(cp):
    rng = np.random.default_rng(1)
    deltas = np.concatenate([rng.normal(-0.03, 0.05, 40), [np.nan, np.nan]])
    out = cp.bootstrap_paired_median(deltas, np.random.default_rng(0), 2000)
    assert out["n_sites"] == 40
    assert out["ci_low"] <= out["median"] <= out["ci_high"]
    assert out["median"] == pytest.approx(np.nanmedian(deltas))
    assert 0.0 <= out["frac_sites_swim_higher"] <= 1.0
    empty = cp.bootstrap_paired_median(np.array([np.nan]), rng, 10)
    assert empty["n_sites"] == 0 and np.isnan(empty["median"])


def test_paired_delta_table_is_deterministic_for_a_seed(cp):
    frame = _metrics_frame(list("abcdefgh"))
    t1 = cp.paired_delta_table(frame, "daily", "all", 500, 7)
    t2 = cp.paired_delta_table(frame, "daily", "all", 500, 7)
    pd.testing.assert_frame_equal(t1, t2)
    assert set(t1["metric"]) == set(cp.METRICS)


def test_transfer_summaries_and_win_rates(cp):
    idx = pd.Index(["a", "b", "c", "d"], name="fid")
    persite = pd.DataFrame(index=idx)
    for key in cp.TRANSFER_CONFIGS:
        for m in cp.TRANSFER_METRICS:
            persite[f"{key}_{m}"] = [0.1, 0.2, 0.3, 0.4]
    persite["ex5_transfer_kge"] = [0.5, 0.1, 0.5, 0.1]
    persite["ex5_transfer_r2"] = [0.5, 0.5, 0.5, np.nan]
    daily = cp.transfer_daily_summary(persite).set_index("config")
    assert len(daily) == 5
    assert daily.loc["Ex5 transferred", "kge_med"] == pytest.approx(0.3)
    monthly = {
        "Ex5 transferred": pd.DataFrame({"kge": [0.6, 0.7], "r2": [0.5, 0.5]}, index=["a", "b"]),
        "E3 calibrated": pd.DataFrame({"kge": [0.5, 0.9], "r2": [0.4, 0.6]}, index=["a", "b"]),
    }
    msum = cp.transfer_monthly_summary(monthly).set_index("config")
    assert msum.loc["E3 calibrated", "kge_med"] == pytest.approx(0.7)
    assert np.isnan(msum.loc["E3 calibrated", "mae_med"])
    wins = cp.transfer_winrates(persite, monthly).set_index("comparison")
    row = wins.loc["Ex5 transferred vs E3 calibrated"]
    assert row["daily_kge_win"] == pytest.approx(0.5) and row["n_daily"] == 4
    assert row["daily_r2_win"] == pytest.approx(1.0)  # r2 NaN at site d drops it
    assert row["monthly_kge_win"] == pytest.approx(0.5) and row["n_monthly"] == 2
    assert np.isnan(wins.loc["Ex5 transferred vs LS ensemble"].get("monthly_kge_win", np.nan))


def test_suffix_frame_drops_ambiguous_bare_columns(cp):
    frame = pd.DataFrame({"kge": [1.0], "kge_swim": [0.7], "r2_swim": [0.5], "kge_rs": [0.8]})
    out = cp._suffix_frame(frame, "swim")
    assert list(out.columns) == ["kge", "r2"]
    assert out["kge"].iloc[0] == 0.7


def test_baseline_comparison_summary_paired_rows(cp):
    comp = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="fid"))
    for name in ("grassbasis_swim", "baseline_it3_swim", "grassbasis_rs", "baseline_rs"):
        for k in cp.METRICS:
            comp[f"{k}_{name}"] = [0.5, 0.6, 0.7]
    comp["kge_grassbasis_swim"] = [0.6, 0.7, 0.9]
    out = cp.baseline_comparison_summary(comp).set_index("series")
    assert out.loc["grassbasis_swim_minus_baseline_it3_swim_paired", "kge_median"] == pytest.approx(
        0.1
    )
    assert "grassbasis_swim_minus_baseline_it4_swim_paired" not in out.index
    assert (out["basis"] == "daily").all()


def test_irrigation_counts_restrict_to_pool(cp):
    tr = pd.DataFrame(
        {
            "site": ["a", "a", "b", "b", "c", "c"],
            "year": [2013, 2014] * 3,
            "irrigated_corrected": [1, 0, 0, 0, 1, 1],
        }
    )
    out = cp.irrigation_counts(tr, ["a", "b"])
    assert out == {
        "n_sites": 2,
        "n_site_years_modelled": 4,
        "ever_irrigated_sites": 1,
        "ever_irrigated_site_ids": ["a"],
        "irrigated_site_years": 1,
    }


def test_pooled_from_timeseries_matches_direct_computation(cp, tmp_path):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2015-01-01", periods=400, freq="D")
    truth = {}
    for fid in ("US-A", "DE-B"):
        flux = pd.Series(rng.uniform(0.5, 5, len(dates)), index=dates)
        flux.iloc[::7] = np.nan  # flux gaps
        swim = flux.fillna(0) * 0.9 + rng.normal(0, 0.2, len(dates))
        rs = flux.fillna(0) * 1.1 + rng.normal(0, 0.2, len(dates))
        rs.iloc[100:105] = np.nan  # a benchmark gap
        pd.DataFrame({"flux_ET": flux, "swim_ET": swim, "benchmark_ET": rs}).rename_axis(
            "date"
        ).to_csv(tmp_path / f"{fid}.csv")
        m = flux.notna() & rs.notna()
        truth[fid] = (flux[m], swim[m], rs[m])
    out = cp.pooled_from_timeseries(tmp_path, ["US-A", "DE-B"], monthly=False).set_index("model")
    obs = np.concatenate([t[0].to_numpy() for t in truth.values()])
    sv = np.concatenate([t[1].to_numpy() for t in truth.values()])
    assert out.loc["swim", "n_points"] == len(obs) and out.loc["swim", "n_stations"] == 2
    assert out.loc["swim", "bias_pooled"] == pytest.approx(float(np.mean(sv - obs)))
    assert out.loc["swim", "r2_pooled"] == pytest.approx(float(np.corrcoef(obs, sv)[0, 1] ** 2))
    counts = np.array([len(t[0]) for t in truth.values()])
    mbe = np.array([float(np.mean(t[1] - t[0])) for t in truth.values()])
    assert out.loc["swim", "mbe_weighted"] == pytest.approx(
        float(np.sum(mbe * np.sqrt(counts)) / np.sum(np.sqrt(counts)))
    )
    monthly = cp.pooled_from_timeseries(tmp_path, ["US-A", "DE-B"], monthly=True).set_index("model")
    assert monthly.loc["swim", "n_stations"] == 2
    assert monthly.loc["swim", "n_points"] <= 2 * 14
    assert monthly.loc["swim", "mean_obs"] > out.loc["swim", "mean_obs"]
