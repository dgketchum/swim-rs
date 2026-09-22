"""Unit tests for the E2 post-calibration archive and evaluation-summary helpers.

Covers the pure pieces of ``archive_postcalibration.py`` (PEST parameter-name
decoding, per-batch posterior merge, medians, per-site bounds and boundary-hit rates, phi-history
checks, ingested-vs-posterior comparison) and of ``evaluation_summary.py``
(metrics with KGE components against the evaluator's ``calc_metrics``, the declared daily and
monthly pairing masks, retrieval-day masks and the overpass split, count reconciliation, grouped
medians, and the NaN-aware reproduction delta).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EX6_DIR = REPO_ROOT / "examples" / "6_Flux_International"


def _load(name):
    sys.path.insert(0, str(EX6_DIR))
    spec = importlib.util.spec_from_file_location(name, EX6_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(EX6_DIR)]
    return module


@pytest.fixture(scope="module")
def arch():
    return _load("archive_postcalibration")


@pytest.fixture(scope="module")
def summ():
    return _load("evaluation_summary")


def _pcol(param, fid):
    return f"pname:p_{param}_{fid}_:0_ptype:cn_usecol:1_pstyle:m"


# ---------------------------------------------------------------------------
# archive helpers
# ---------------------------------------------------------------------------


def test_decode_par_column_handles_underscored_params_and_hyphenated_fids(arch):
    assert arch.decode_par_column(_pcol("ndvi_0", "US-Tw3")) == ("US-Tw3", "ndvi_0")
    assert arch.decode_par_column(_pcol("swe_alpha", "AU-Rgf")) == ("AU-Rgf", "swe_alpha")
    assert arch.decode_par_column(_pcol("aw", "IT-CA2")) == ("IT-CA2", "aw")
    assert arch.decode_par_column("real_name") is None


def _write_par(path, fids, params, n_real=5, seed=0):
    rng = np.random.default_rng(seed)
    cols = [_pcol(p, f) for f in fids for p in params]
    data = rng.uniform(0.2, 0.8, size=(n_real + 1, len(cols)))
    idx = [str(i) for i in range(n_real)] + ["base"]
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.index.name = "real_name"
    df.to_csv(path)
    return df


def test_merge_par_csvs_joins_batches_and_posterior_medians_exclude_base(arch, tmp_path):
    a = _write_par(tmp_path / "a.par.csv", ["US-A", "US-B"], ["aw", "mad"], seed=1)
    b = _write_par(tmp_path / "b.par.csv", ["DE-C"], ["aw", "mad"], seed=2)
    merged = arch.merge_par_csvs([tmp_path / "a.par.csv", tmp_path / "b.par.csv"])
    assert merged.shape == (6, 6)
    assert "base" in merged.index

    med = arch.posterior_medians(merged, ["US-A", "US-B", "DE-C"])
    assert list(med.index) == ["DE-C", "US-A", "US-B"]
    expect = a.loc[a.index != "base", _pcol("aw", "US-A")].median()
    assert med.loc["US-A", "aw"] == pytest.approx(expect)
    expect_b = b.loc[b.index != "base", _pcol("mad", "DE-C")].median()
    assert med.loc["DE-C", "mad"] == pytest.approx(expect_b)


def test_merge_par_csvs_rejects_duplicate_site_parameters(arch, tmp_path):
    _write_par(tmp_path / "a.par.csv", ["US-A"], ["aw"])
    _write_par(tmp_path / "b.par.csv", ["US-A"], ["aw"])
    with pytest.raises(ValueError, match="more than one batch"):
        arch.merge_par_csvs([tmp_path / "a.par.csv", tmp_path / "b.par.csv"])


def test_posterior_medians_raises_on_unknown_site(arch, tmp_path):
    _write_par(tmp_path / "a.par.csv", ["US-A", "US-Z"], ["aw"])
    merged = arch.merge_par_csvs([tmp_path / "a.par.csv"])
    with pytest.raises(KeyError):
        arch.posterior_medians(merged, ["US-A"])


def _write_par_data(path, rows):
    pd.DataFrame(
        [(_pcol(p, f), lo, hi) for f, p, lo, hi in rows], columns=["parnme", "parlbnd", "parubnd"]
    ).to_csv(path, index=False)


def test_param_bounds_are_per_site_and_boundary_hits_use_each_sites_own_bounds(arch, tmp_path):
    # mad is bounded 0.3-0.8 on rainfed sites and 0.1-0.3 on irrigated sites
    _write_par_data(
        tmp_path / "pd0.csv",
        [("US-R", "mad", 0.3, 0.8), ("US-I", "mad", 0.1, 0.3), ("US-R", "aw", 100, 400)],
    )
    _write_par_data(tmp_path / "pd1.csv", [("US-I", "aw", 100, 400)])
    bounds = arch.param_bounds([tmp_path / "pd0.csv", tmp_path / "pd1.csv"], ["US-R", "US-I"])
    assert bounds.loc[("US-I", "mad"), "upper"] == 0.3
    assert bounds.loc[("US-R", "mad"), "lower"] == 0.3

    med = pd.DataFrame(
        {"mad": [0.3, 0.3], "aw": [399.0, 250.0]}, index=pd.Index(["US-R", "US-I"], name="site_id")
    )
    rows = {r["parameter"]: r for r in arch.boundary_hit_rows(med, bounds, "ALL", "all", "run")}
    # 0.3 is US-R's lower bound and US-I's upper bound; a group-level bound would miss one
    assert rows["mad"]["lower_hit_rate"] == 0.5
    assert rows["mad"]["upper_hit_rate"] == 0.5
    assert rows["mad"]["bounds"] == "0.1-0.3;0.3-0.8"
    assert rows["mad"]["n_sites"] == 2
    assert rows["aw"]["upper_hit_rate"] == 0.5
    assert rows["aw"]["boundary_seeking"] is False
    assert rows["aw"]["internal_name"] == "aw"


def test_boundary_hit_rows_maps_pest_names_and_flags_railing(arch):
    bounds = pd.DataFrame(
        {"lower": [0.01, 0.01, 0.01], "upper": [1.0, 1.0, 1.0]},
        index=pd.MultiIndex.from_tuples(
            [("a", "ks_alpha"), ("b", "ks_alpha"), ("c", "ks_alpha")], names=["site_id", "param"]
        ),
    )
    med = pd.DataFrame({"ks_alpha": [0.01, 0.015, 0.5]}, index=["a", "b", "c"])
    (row,) = arch.boundary_hit_rows(med, bounds, "rainfed", "irrigation_class", "run")
    assert row["internal_name"] == "ks_damp"
    assert row["lower_hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["boundary_seeking"] is True
    assert row["group_kind"] == "irrigation_class"


def test_param_bounds_rejects_bounds_in_two_batches(arch, tmp_path):
    _write_par_data(tmp_path / "pd0.csv", [("US-R", "aw", 100, 400)])
    _write_par_data(tmp_path / "pd1.csv", [("US-R", "aw", 100, 400)])
    with pytest.raises(ValueError, match="more than one batch"):
        arch.param_bounds([tmp_path / "pd0.csv", tmp_path / "pd1.csv"], ["US-R"])


def test_phi_history_checks(arch):
    assert arch.phi_history_checks([305143.0, 26868.0, 21510.0, 20841.0]) == []
    assert any("strictly decreasing" in p for p in arch.phi_history_checks([10.0, 5.0, 6.0, 4.0]))
    assert any("not below" in p for p in arch.phi_history_checks([5.0, 6.0]))
    assert arch.phi_history_checks([1.0, np.nan]) == ["phi history empty or non-finite"]
    assert arch.phi_history_checks([]) == ["phi history empty or non-finite"]


def test_compare_ingested_maps_internal_names_and_reports_relative_error(arch):
    medians = pd.DataFrame({"ks_alpha": [0.4], "aw": [300.0]}, index=["US-A"])
    container = pd.DataFrame({"ks_damp": [0.4], "aw": [303.0]}, index=["US-A"])
    out = arch.compare_ingested(container, medians).set_index("pest_param")
    assert out.loc["ks_alpha", "internal"] == "ks_damp"
    assert out.loc["ks_alpha", "rel_err"] == pytest.approx(0.0)
    assert out.loc["aw", "rel_err"] == pytest.approx(0.01)

    missing = arch.compare_ingested(container, pd.DataFrame({"aw": [1.0]}, index=["US-Z"]))
    assert np.isnan(missing["container"].iloc[0]) and np.isnan(missing["rel_err"].iloc[0])


# ---------------------------------------------------------------------------
# evaluation-summary helpers
# ---------------------------------------------------------------------------


def test_full_metrics_matches_evaluator_calc_metrics_and_adds_components(summ):
    sys.path.insert(0, str(EX6_DIR))
    try:
        import evaluate as ev
    finally:
        sys.path.remove(str(EX6_DIR))
    rng = np.random.default_rng(3)
    obs = rng.uniform(0.5, 6.0, 60)
    mod = obs * 1.1 + rng.normal(0, 0.4, 60)
    mod[[3, 17]] = np.nan
    ref = ev.calc_metrics(obs, mod)
    got = summ.full_metrics(obs, mod)
    assert got["n"] == ref["n"] == 58
    for k in ("r2", "r", "rmse", "bias", "kge"):
        assert got[k] == pytest.approx(ref[k], abs=1e-10)
    o, m = obs[np.isfinite(mod)], mod[np.isfinite(mod)]
    assert got["alpha"] == pytest.approx(np.std(m) / np.std(o))
    assert got["beta"] == pytest.approx(np.mean(m) / np.mean(o))
    assert got["mae"] == pytest.approx(np.mean(np.abs(m - o)))
    assert got["kge"] == pytest.approx(
        1 - np.sqrt((got["r"] - 1) ** 2 + (got["alpha"] - 1) ** 2 + (got["beta"] - 1) ** 2)
    )


def test_full_metrics_below_minimum_is_nan_but_counts(summ):
    got = summ.full_metrics(np.arange(9.0), np.arange(9.0))
    assert got["n"] == 9
    assert all(np.isnan(got[k]) for k in summ.METRICS)
    got6 = summ.full_metrics(np.arange(6.0), np.arange(6.0), min_n=6)
    assert got6["r2"] == pytest.approx(1.0)


def _series(n=400, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    flux = pd.Series(rng.uniform(0.5, 5.0, n), idx)
    swim = flux * 1.05
    rs = flux * 0.9
    return flux, swim, rs


def test_daily_pairing_declares_reasons_and_uses_the_triple_finite_mask(summ):
    flux, swim, rs = _series()
    flux.iloc[:50] = np.nan
    rs.iloc[50:70] = np.nan
    reason, dates, obs, sv, rv = summ.daily_pairing(flux, swim, rs)
    assert reason is None
    assert len(dates) == 400 - 70
    assert np.isfinite(obs).all() and np.isfinite(rv).all()

    reason, *_ = summ.daily_pairing(flux.iloc[:5], swim, rs)
    assert reason == "overlap_lt_10_days"
    reason, *_ = summ.daily_pairing(flux, swim, rs * np.nan)
    assert reason == "paired_days_lt_10"


def test_monthly_pairing_requires_six_paired_months_and_full_benchmark_months(summ):
    flux, swim, rs = _series(n=400)
    reason, months, obs, sv, rv = summ.monthly_pairing(flux, swim, rs)
    assert reason is None
    assert len(months) >= 12
    assert obs.sum() == pytest.approx(
        flux.loc[months[0] : months[-1] + pd.offsets.MonthEnd(0)].sum()
    )
    # one missing benchmark day removes that whole month from the paired set
    rs2 = rs.copy()
    rs2.loc["2020-03-15"] = np.nan
    _, months2, *_ = summ.monthly_pairing(flux, swim, rs2)
    assert pd.Timestamp("2020-03-01") in months and pd.Timestamp("2020-03-01") not in months2
    reason, *_ = summ.monthly_pairing(flux.iloc[:20], swim, rs)
    assert reason == "daily_overlap_lt_30"
    reason, *_ = summ.monthly_pairing(flux.iloc[:120], swim, rs)
    assert reason == "paired_months_lt_6"


def test_retrieval_mask_and_overpass_split_row(summ):
    flux, swim, rs = _series(n=200)
    dates = flux.index
    m1 = pd.Series(0.5, index=dates[::16])
    m2 = pd.Series(0.6, index=dates[8::16])
    m2.iloc[0] = np.nan
    is_ret = summ.retrieval_mask(dates, [m1, m2])
    assert is_ret.sum() == len(m1) + len(m2) - 1
    row = summ.overpass_split_row(
        "X", dates, flux.to_numpy(), swim.to_numpy(), rs.to_numpy(), is_ret
    )
    assert row["n_retrieval"] + row["n_between"] == row["n_paired"] == 200
    assert row["kge_swim_minus_rs_retrieval"] == pytest.approx(
        row["kge_swim_retrieval"] - row["kge_rs_retrieval"]
    )
    assert row["r2_support_interaction"] == pytest.approx(
        row["r2_swim_minus_rs_between"] - row["r2_swim_minus_rs_retrieval"]
    )
    few = np.zeros(200, dtype=bool)
    few[:5] = True
    assert (
        summ.overpass_split_row("X", dates, flux.to_numpy(), swim.to_numpy(), rs.to_numpy(), few)
        is None
    )


def test_reconcile_counts(summ):
    excl = pd.DataFrame({"site": ["a", "b", "c"], "reason": ["no_flux_data"] * 3})
    assert summ.reconcile_counts(66, 63, excl)["reconciles"] is True
    assert summ.reconcile_counts(66, 62, excl)["reconciles"] is False


def test_group_medians_reports_n_and_medians_per_group(summ):
    metrics = pd.DataFrame(
        {"kge_swim": [0.1, 0.3, 0.5], "kge_rs": [0.2, 0.2, 0.2], "n": [10, 10, 10]},
        index=pd.Index(["a", "b", "c"], name="fid"),
    )
    groups = pd.DataFrame(
        {"region": ["CONUS", "CONUS", "ex-CONUS"]}, index=pd.Index(["a", "b", "c"], name="site_id")
    )
    out = summ.group_medians(metrics, groups, "daily").set_index("group")
    assert out.loc["CONUS", "n_sites"] == 2
    assert out.loc["CONUS", "kge_swim_median"] == pytest.approx(0.2)
    assert out.loc["ex-CONUS", "kge_swim_median"] == pytest.approx(0.5)
    assert "n_median" not in out.columns


def test_repro_delta_is_nan_aware(summ):
    keys = [f"{k}_{s}" for k in ("r2", "r", "rmse", "bias", "kge") for s in ("swim", "rs")]
    ev_row = pd.Series({k: 0.5 for k in keys})
    row = {k: 0.5 for k in keys}
    row["kge_rs"] = 0.5 + 1e-3
    assert summ._repro_delta(ev_row, row) == pytest.approx(1e-3)
    ev_row["r_swim"] = np.nan
    assert summ._repro_delta(ev_row, row) == float("inf")
    row["r_swim"] = np.nan
    assert summ._repro_delta(ev_row, row) == pytest.approx(1e-3)
