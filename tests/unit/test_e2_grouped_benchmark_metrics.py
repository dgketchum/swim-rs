"""Unit tests for the Example 5 grouped SWIM-OpenET benchmark estimators.

Covers the pure grouped-aggregation primitives added to
``examples/5_Flux_Ensemble/evaluate.py`` under the VALIDATION_POLICY
"SWIM-OpenET Benchmark Aggregation" contract:

- pooled KGE/RMSE/MBE/r/r^2/slope0 on concatenated exactly-paired cohorts;
- sqrt(n)-weighted site KGE/RMSE/MBE;
- moment-based sufficient-statistic implementation vs direct concatenation;
- deterministic whole-site paired bootstrap with shared draws for both models;
- descriptive failures on degenerate inputs;
- grouped artifact schema, deterministic ordering, and DIY name separation.

The script lives outside the swimrs package and is imported by path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble" / "evaluate.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("e2_evaluate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _support(n):
    """Deterministic support-class mix: captures every 8th day, flat-fill tail."""
    out = []
    for i in range(n):
        if i % 8 == 0:
            out.append("capture")
        elif i >= n - 3:
            out.append("flat_fill")
        else:
            out.append("interpolated")
    return tuple(out)


def _record(ev, fid, n, seed, bias=0.0, noise=0.4, scale_mod=1.0, start="2020-01-01", support=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    obs = np.abs(rng.normal(3.5, 1.2, size=n)) + 0.5
    swim = obs * scale_mod + bias + rng.normal(0.0, noise, size=n)
    openet = obs + rng.normal(0.0, noise, size=n) - 0.3
    return ev.PairedSiteSeries(
        fid=fid,
        index=idx,
        observed=obs,
        swim=swim,
        openet=openet,
        support_class=_support(n) if support else None,
    )


@pytest.fixture(scope="module")
def two_site_cohort(ev):
    # deliberately unequal lengths and different error structure per site
    a = _record(ev, "US-Aaa", 400, seed=1, bias=0.8, scale_mod=1.2)
    b = _record(ev, "US-Bbb", 40, seed=2, bias=-0.9, scale_mod=0.7)
    return (a, b)


@pytest.fixture(scope="module")
def multi_site_cohort(ev):
    return (
        _record(ev, "US-Aaa", 311, seed=11, bias=0.4, scale_mod=1.1),
        _record(ev, "US-Bbb", 57, seed=12, bias=-0.6, scale_mod=0.8),
        _record(ev, "US-Ccc", 1200, seed=13, bias=0.05, scale_mod=1.0),
        _record(ev, "US-Ddd", 23, seed=14, bias=-1.2, scale_mod=1.3),
    )


# ---------------------------------------------------------------------------
# Constants and definitions
# ---------------------------------------------------------------------------


def test_constants(ev):
    assert ev.MIN_OBS_FOR_METRICS == 10
    assert ev.BOOTSTRAP_REPS_DEFAULT == 10_000
    assert ev.BOOTSTRAP_SEED_DEFAULT == 42
    assert ev.PRIMARY_METRICS == ("kge", "rmse", "mbe")
    assert ev.POOLED_METRICS == ("kge", "rmse", "mbe", "r", "r2", "slope0")
    assert ev.WEIGHTED_METRICS == ("kge", "rmse", "mbe")


# ---------------------------------------------------------------------------
# Case 1-2: pooled vs weighted vs site mean/median are distinct estimands
# ---------------------------------------------------------------------------


def test_pooled_kge_differs_from_site_aggregates(ev, two_site_cohort):
    est = ev.grouped_point_estimates(two_site_cohort)
    pooled_kge = est[(ev.AGG_POOLED, "swim", "kge")]
    weighted_kge = est[(ev.AGG_WEIGHTED, "swim", "kge")]

    site_kge = ev.site_metric_triads(two_site_cohort, "swim")["kge"]
    mean_kge = site_kge.mean()
    median_kge = np.median(site_kge)

    # KGE is nonlinear: the pooled value is not any site-level aggregate
    for other in (mean_kge, median_kge, weighted_kge):
        assert abs(pooled_kge - other) > 1e-3
    # both aggregations are emitted under distinct keys
    assert (ev.AGG_POOLED, "swim", "kge") in est
    assert (ev.AGG_WEIGHTED, "swim", "kge") in est


def test_weighted_rmse_mbe_differ_from_pooled_and_unweighted(ev, two_site_cohort):
    est = ev.grouped_point_estimates(two_site_cohort)
    triads = ev.site_metric_triads(two_site_cohort, "swim")
    for q in ("rmse", "mbe"):
        pooled = est[(ev.AGG_POOLED, "swim", q)]
        weighted = est[(ev.AGG_WEIGHTED, "swim", q)]
        unweighted_mean = triads[q].mean()
        assert abs(pooled - weighted) > 1e-3
        assert abs(weighted - unweighted_mean) > 1e-3


def test_weighted_formula_exact(ev, two_site_cohort):
    n = np.array([r.n for r in two_site_cohort], dtype=float)
    w = np.sqrt(n)
    triads = ev.site_metric_triads(two_site_cohort, "swim")
    wm = ev.sqrt_n_weighted_metrics(triads, n)
    for q in ev.WEIGHTED_METRICS:
        expected = np.sum(w * triads[q]) / np.sum(w)
        assert wm[q] == pytest.approx(expected, abs=1e-14)


# ---------------------------------------------------------------------------
# Case 3: signed MBE preserved in both aggregations
# ---------------------------------------------------------------------------


def test_signed_mbe_preserved(ev):
    pos = _record(ev, "US-Pos", 60, seed=3, bias=1.5, noise=0.05)
    neg = _record(ev, "US-Neg", 60, seed=4, bias=-1.5, noise=0.05)
    triads = ev.site_metric_triads((pos, neg), "swim")
    assert triads["mbe"][0] > 1.0
    assert triads["mbe"][1] < -1.0
    est = ev.grouped_point_estimates((pos, neg))
    # equal-length sites with opposite bias: pooled and weighted MBE near zero,
    # but signed (never folded to absolute values)
    assert abs(est[(ev.AGG_POOLED, "swim", "mbe")]) < 0.5
    assert abs(est[(ev.AGG_WEIGHTED, "swim", "mbe")]) < 0.5
    single = ev.grouped_point_estimates((neg,))
    assert single[(ev.AGG_POOLED, "swim", "mbe")] < -1.0
    assert single[(ev.AGG_WEIGHTED, "swim", "mbe")] < -1.0


# ---------------------------------------------------------------------------
# Case 4: moment-based implementation reproduces direct concatenation
# ---------------------------------------------------------------------------


def test_moment_identity_all_six_metrics(ev, multi_site_cohort):
    for model in ("swim", "openet_ensemble"):
        moments = np.sum(
            [
                ev.site_sufficient_stats(r.observed, r.model_series(model))
                for r in multi_site_cohort
            ],
            axis=0,
        )
        from_moments = ev.pooled_metrics_from_moments(moments, context="test")
        obs = np.concatenate([r.observed for r in multi_site_cohort])
        mod = np.concatenate([r.model_series(model) for r in multi_site_cohort])
        direct = ev.pooled_metrics_direct(obs, mod)
        for k in ev.POOLED_METRICS:
            assert from_moments[k] == pytest.approx(direct[k], abs=1e-12), (model, k)


def test_pooled_metrics_runtime_identity_gate(ev, multi_site_cohort):
    # the canonical path runs the identity assertion internally and returns
    pm = ev.pooled_metrics(multi_site_cohort, "swim")
    assert set(pm) >= set(ev.POOLED_METRICS)


# ---------------------------------------------------------------------------
# Case 5-6: r^2 and slope0 definitions
# ---------------------------------------------------------------------------


def test_r2_is_squared_pearson_not_nse(ev):
    rec = _record(ev, "US-Bias", 200, seed=5, bias=2.0, noise=0.1)
    pm = ev.pooled_metrics_direct(rec.observed, rec.swim)
    assert pm["r2"] == pytest.approx(pm["r"] ** 2, abs=1e-14)
    assert 0.0 <= pm["r2"] <= 1.0
    nse = r2_score(rec.observed, rec.swim)
    # heavy bias: NSE is strongly degraded while Pearson r^2 stays high
    assert abs(pm["r2"] - nse) > 0.5


def test_slope0_zero_intercept_definition(ev):
    rng = np.random.default_rng(6)
    obs = np.linspace(1.0, 8.0, 120)
    mod = 0.8 * obs + 1.5 + rng.normal(0, 0.05, size=120)  # nonzero intercept
    pm = ev.pooled_metrics_direct(obs, mod)
    expected = np.sum(obs * mod) / np.sum(obs * obs)
    assert pm["slope0"] == pytest.approx(expected, abs=1e-14)
    fitted_slope = np.polyfit(obs, mod, 1)[0]
    assert abs(pm["slope0"] - fitted_slope) > 0.05


# ---------------------------------------------------------------------------
# Case 7: exact common support shared by both models
# ---------------------------------------------------------------------------


def test_common_support_mask_shared_across_models(ev):
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    flux = np.full(30, 3.0)
    swim = np.full(30, 3.1)
    openet = np.full(30, 2.9)
    flux += np.linspace(0, 1, 30)  # avoid constant series
    swim += np.linspace(0, 1.1, 30)
    openet += np.linspace(0, 0.9, 30)
    swim[5] = np.nan  # a SWIM hole must also remove the day from OpenET
    openet[7] = np.nan  # and vice versa
    rec = ev.build_paired_site_series("US-Msk", idx, flux, swim, openet)
    assert rec.n == 28
    assert len(rec.observed) == len(rec.swim) == len(rec.openet) == 28
    assert pd.Timestamp("2020-01-06") not in rec.index
    assert pd.Timestamp("2020-01-08") not in rec.index


def test_paired_series_rejects_nonfinite_after_mask(ev):
    idx = pd.date_range("2020-01-01", periods=12, freq="D")
    good = np.linspace(1, 2, 12)
    bad = good.copy()
    bad[3] = np.nan
    with pytest.raises(ev.GroupedEstimationError, match="non-finite"):
        ev.PairedSiteSeries(fid="US-Bad", index=idx, observed=good, swim=bad, openet=good)


# ---------------------------------------------------------------------------
# Case 8: site order invariance
# ---------------------------------------------------------------------------


def test_site_order_invariance(ev, multi_site_cohort):
    est = ev.grouped_point_estimates(multi_site_cohort)
    est_rev = ev.grouped_point_estimates(tuple(reversed(multi_site_cohort)))
    assert set(est) == set(est_rev)
    for k in est:
        assert est[k] == pytest.approx(est_rev[k], abs=1e-12), k


# ---------------------------------------------------------------------------
# Case 9-11: bootstrap determinism, shared draws, duplicated-site blocks
# ---------------------------------------------------------------------------


def test_bootstrap_deterministic_under_seed(ev, multi_site_cohort):
    b1 = ev.bootstrap_grouped(multi_site_cohort, reps=200, seed=42)
    b2 = ev.bootstrap_grouped(multi_site_cohort, reps=200, seed=42)
    assert set(b1) == set(b2)
    for k in b1:
        np.testing.assert_array_equal(b1[k], b2[k])
    b3 = ev.bootstrap_grouped(multi_site_cohort, reps=200, seed=7)
    assert not np.array_equal(
        b1[(ev.AGG_POOLED, "swim", "kge")], b3[(ev.AGG_POOLED, "swim", "kge")]
    )


def test_bootstrap_shared_draws_for_both_models(ev):
    # identical model series: every contrast replicate must be exactly zero,
    # which can only happen if both models share the same site multiplicities
    recs = []
    for i, n in enumerate((45, 120, 15)):
        rng = np.random.default_rng(20 + i)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        obs = np.abs(rng.normal(3.0, 1.0, size=n)) + 0.5
        mod = obs + rng.normal(0, 0.3, size=n)
        recs.append(
            ev.PairedSiteSeries(fid=f"US-{i}", index=idx, observed=obs, swim=mod, openet=mod)
        )
    boot = ev.bootstrap_grouped(tuple(recs), reps=100, seed=42)
    for agg, metrics in (
        (ev.AGG_POOLED, ev.POOLED_METRICS),
        (ev.AGG_WEIGHTED, ev.WEIGHTED_METRICS),
    ):
        for k in metrics:
            np.testing.assert_allclose(
                boot[(agg, "swim_minus_openet", k)], 0.0, atol=1e-12, err_msg=f"{agg}/{k}"
            )


def test_duplicated_site_contributes_full_block_and_weight(ev, two_site_cohort):
    reps = 50
    idx_matrix, counts = ev._bootstrap_multiplicities(len(two_site_cohort), reps, seed=42)
    boot = ev.bootstrap_grouped(two_site_cohort, reps=reps, seed=42)
    # independently rebuild replicate 0 by physically concatenating the drawn
    # site blocks with their multiplicities
    draw = idx_matrix[0]
    obs = np.concatenate([two_site_cohort[i].observed for i in draw])
    mod = np.concatenate([two_site_cohort[i].swim for i in draw])
    direct = ev.pooled_metrics_direct(obs, mod)
    for k in ev.POOLED_METRICS:
        assert boot[(ev.AGG_POOLED, "swim", k)][0] == pytest.approx(direct[k], abs=1e-9), k
    # weighted: duplicated sites carry their sqrt(n) weight with multiplicity
    n = np.array([r.n for r in two_site_cohort], dtype=float)
    w = np.sqrt(n)
    triads = ev.site_metric_triads(two_site_cohort, "swim")
    for q in ev.WEIGHTED_METRICS:
        expected = np.sum(counts[0] * w * triads[q]) / np.sum(counts[0] * w)
        assert boot[(ev.AGG_WEIGHTED, "swim", q)][0] == pytest.approx(expected, abs=1e-12), q


# ---------------------------------------------------------------------------
# Case 12: degenerate inputs fail with descriptive errors
# ---------------------------------------------------------------------------


def test_constant_observations_raise(ev):
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    obs = np.full(20, 2.0)
    mod = np.linspace(1, 3, 20)
    rec = ev.PairedSiteSeries(fid="US-Const", index=idx, observed=obs, swim=mod, openet=mod)
    with pytest.raises(ev.GroupedEstimationError, match="variance"):
        ev.pooled_metrics((rec,), "swim")


def test_zero_observed_mean_raises(ev):
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    obs = np.tile([-2.0, 2.0], 10)  # mean exactly zero
    mod = np.linspace(0, 2, 20)
    rec = ev.PairedSiteSeries(fid="US-Zm", index=idx, observed=obs, swim=mod, openet=mod)
    with pytest.raises(ev.GroupedEstimationError, match="mean"):
        ev.pooled_metrics((rec,), "swim")


def test_duplicate_dates_raise(ev):
    idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-02"] + [f"2020-01-{d:02d}" for d in range(3, 12)]
    )
    n = len(idx)
    vals = np.linspace(1, 2, n)
    with pytest.raises(ev.GroupedEstimationError, match="duplicate"):
        ev.PairedSiteSeries(fid="US-Dup", index=idx, observed=vals, swim=vals, openet=vals)


def test_below_min_obs_raises(ev):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    vals = np.linspace(1, 2, 5)
    with pytest.raises(ev.GroupedEstimationError, match="MIN_OBS_FOR_METRICS"):
        ev.PairedSiteSeries(fid="US-Few", index=idx, observed=vals, swim=vals, openet=vals)


def test_empty_cohort_raises(ev):
    with pytest.raises(ev.GroupedEstimationError, match="empty"):
        ev.grouped_point_estimates(())


def test_duplicate_site_id_raises(ev, two_site_cohort):
    with pytest.raises(ev.GroupedEstimationError, match="duplicate"):
        ev.grouped_point_estimates((two_site_cohort[0], two_site_cohort[0]))


# ---------------------------------------------------------------------------
# Case 13: reps=0 development mode leaves CI fields null
# ---------------------------------------------------------------------------


def test_zero_reps_yields_null_ci(ev, two_site_cohort):
    metrics, contrasts = ev.grouped_metric_tables(two_site_cohort, "daily", reps=0, seed=42)
    assert metrics["estimate"].notna().all()
    assert metrics["ci95_low"].isna().all()
    assert metrics["ci95_high"].isna().all()
    assert (metrics["bootstrap_reps"] == 0).all()
    assert contrasts["ci95_low"].isna().all()


# ---------------------------------------------------------------------------
# Grouped artifact schema, ordering, uniqueness
# ---------------------------------------------------------------------------


def test_grouped_metrics_schema_and_order(ev, two_site_cohort):
    metrics, contrasts = ev.grouped_metric_tables(two_site_cohort, "daily", reps=100, seed=42)
    assert list(metrics.columns) == list(ev.GROUPED_METRIC_COLUMNS)
    assert list(contrasts.columns) == list(ev.GROUPED_CONTRAST_COLUMNS)
    # 18 rows: 6 pooled x 2 models + 3 weighted x 2 models
    assert len(metrics) == 18
    assert len(contrasts) == 9
    key = metrics[["scale", "aggregation", "model", "metric"]].apply(tuple, axis=1)
    assert key.is_unique
    # deterministic sort: pooled before weighted, swim before openet_ensemble,
    # metric in declared order
    expected = []
    for agg, mets in ((ev.AGG_POOLED, ev.POOLED_METRICS), (ev.AGG_WEIGHTED, ev.WEIGHTED_METRICS)):
        for model in ("swim", "openet_ensemble"):
            for k in mets:
                expected.append(("daily", agg, model, k))
    assert list(key) == expected
    # aggregation-specific annotations
    pooled = metrics[metrics["aggregation"] == ev.AGG_POOLED]
    weighted = metrics[metrics["aggregation"] == ev.AGG_WEIGHTED]
    assert (pooled["weight_rule"] == "none").all()
    assert (weighted["weight_rule"] == "sqrt(n_site)").all()
    kge_rows = metrics[metrics["metric"] == "kge"]
    assert (kge_rows["kge_variant"] == "2009").all()
    r2_rows = metrics[metrics["metric"] == "r2"]
    assert (r2_rows["r2_definition"] == "pearson_r_squared").all()
    slope_rows = metrics[metrics["metric"] == "slope0"]
    assert (slope_rows["slope_constraint"] == "intercept_forced_zero").all()
    assert (metrics["bootstrap_unit"] == "site").all()
    # units
    err_rows = metrics[metrics["metric"].isin(["rmse", "mbe"])]
    assert (err_rows["unit"] == "mm d-1").all()
    dim_rows = metrics[~metrics["metric"].isin(["rmse", "mbe"])]
    assert (dim_rows["unit"] == "dimensionless").all()


def test_contrast_directions(ev, two_site_cohort):
    _, contrasts = ev.grouped_metric_tables(two_site_cohort, "monthly", reps=0, seed=42)
    assert (contrasts["contrast"] == "swim_minus_openet").all()
    direction = dict(
        zip(contrasts["metric"] + "|" + contrasts["aggregation"], contrasts["favorable_direction"])
    )
    for agg in (ev.AGG_POOLED, ev.AGG_WEIGHTED):
        assert direction[f"kge|{agg}"] == "positive"
        assert direction[f"rmse|{agg}"] == "negative"
        assert direction[f"mbe|{agg}"] == "directional_only"
    assert direction[f"r|{ev.AGG_POOLED}"] == "positive"
    assert direction[f"r2|{ev.AGG_POOLED}"] == "positive"
    assert direction[f"slope0|{ev.AGG_POOLED}"] == "directional_only"
    err_rows = contrasts[contrasts["metric"].isin(["rmse", "mbe"])]
    assert (err_rows["unit"] == "mm month-1").all()


def test_contrast_estimates_match_model_difference(ev, two_site_cohort):
    metrics, contrasts = ev.grouped_metric_tables(two_site_cohort, "daily", reps=0, seed=42)
    m = metrics.set_index(["aggregation", "model", "metric"])["estimate"]
    for _, row in contrasts.iterrows():
        expected = (
            m[(row["aggregation"], "swim", row["metric"])]
            - m[(row["aggregation"], "openet_ensemble", row["metric"])]
        )
        assert row["estimate"] == pytest.approx(expected, abs=1e-14)


def test_ci_brackets_estimate(ev, multi_site_cohort):
    metrics, _ = ev.grouped_metric_tables(multi_site_cohort, "daily", reps=500, seed=42)
    ok = (metrics["ci95_low"] <= metrics["estimate"]) & (
        metrics["estimate"] <= metrics["ci95_high"]
    )
    # point estimate is the original-cohort statistic; with moderate reps it
    # should almost always fall inside its own site-bootstrap interval
    assert ok.all()


# ---------------------------------------------------------------------------
# Case 14 + artifact writing: site-effect optionality, filenames, metadata
# ---------------------------------------------------------------------------


def _bundle(ev, records, scale="daily", reps=50, site_effect=False):
    metrics, contrasts = ev.grouped_metric_tables(records, scale, reps=reps, seed=42)
    se = ev.site_effect_summary(records, reps=reps, seed=42, scale=scale) if site_effect else None
    site_metrics = pd.DataFrame(
        {"n": [r.n for r in records]}, index=pd.Index([r.fid for r in records], name="fid")
    )
    meta = ev.grouped_metadata(records, scale, reps, 42, "volk", {})
    return ev.BenchmarkEvaluation(
        site_metrics=site_metrics,
        grouped_metrics=metrics,
        grouped_contrasts=contrasts,
        paired_records=tuple(records),
        site_effect_summary=se,
        metadata=meta,
    )


def test_site_effect_absent_by_default(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, site_effect=False)
    paths = ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")
    assert not (tmp_path / "evaluation_site_effect_summary_daily.csv").exists()
    assert (tmp_path / "evaluation_grouped_daily_metrics.csv").exists()
    assert (tmp_path / "evaluation_grouped_daily_contrasts.csv").exists()
    assert (tmp_path / "evaluation_grouped_daily_metadata.json").exists()
    assert "site_effect" not in paths or paths.get("site_effect") is None


def test_site_effect_written_when_requested(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, site_effect=True)
    ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")
    se_path = tmp_path / "evaluation_site_effect_summary_daily.csv"
    assert se_path.exists()
    se = pd.read_csv(se_path)
    assert (se["aggregation"] == "median_paired_site_effect").all()
    assert sorted(se["metric"]) == ["kge", "mbe", "rmse"]


def test_site_effect_flag_does_not_alter_grouped_artifacts(ev, two_site_cohort, tmp_path):
    d1, d2 = tmp_path / "without", tmp_path / "with"
    d1.mkdir()
    d2.mkdir()
    ev.write_grouped_outputs(_bundle(ev, two_site_cohort, site_effect=False), str(d1), "daily")
    ev.write_grouped_outputs(_bundle(ev, two_site_cohort, site_effect=True), str(d2), "daily")
    for name in ("evaluation_grouped_daily_metrics.csv", "evaluation_grouped_daily_contrasts.csv"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_site_effect_median_definition(ev, two_site_cohort):
    se = ev.site_effect_summary(two_site_cohort, reps=100, seed=42, scale="daily")
    ts = ev.site_metric_triads(two_site_cohort, "swim")
    te = ev.site_metric_triads(two_site_cohort, "openet_ensemble")
    for q in ev.WEIGHTED_METRICS:
        row = se[se["metric"] == q].iloc[0]
        assert row["estimate"] == pytest.approx(np.median(ts[q] - te[q]), abs=1e-12)


def test_diy_names_do_not_overwrite_canonical(ev):
    canon = ev.grouped_output_paths("/x", "daily", "volk")
    diy = ev.grouped_output_paths("/x", "daily", "diy")
    assert set(canon.values()).isdisjoint(set(diy.values()))
    for p in diy.values():
        assert "diy" in Path(p).name


def test_grouped_metadata_content(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=50)
    ev.write_grouped_outputs(
        bundle, str(tmp_path), "daily", openet_source="volk", cli_args={"monthly": False}
    )
    with open(tmp_path / "evaluation_grouped_daily_metadata.json") as f:
        meta = json.load(f)
    assert meta["n_sites"] == 2
    assert meta["n_pairs"] == sum(r.n for r in two_site_cohort)
    assert meta["bootstrap"] == {
        "unit": "site",
        "reps": 50,
        "seed": 42,
        "interval": "percentile_2.5_97.5",
    }
    assert meta["kge_variant"] == "2009"
    assert "formulas" in meta and "slope0" in meta["formulas"]
    assert "mask_definition" in meta
    assert [s["fid"] for s in meta["sites"]] == [r.fid for r in two_site_cohort]
    assert "output_hashes" in meta
    assert "git" in meta
    assert meta["cli_args"] == {"monthly": False}


def test_full_precision_csv_roundtrip(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=0)
    ev.write_grouped_outputs(bundle, str(tmp_path), "daily")
    est = ev.grouped_point_estimates(two_site_cohort)
    written = pd.read_csv(tmp_path / "evaluation_grouped_daily_metrics.csv")
    for _, row in written.iterrows():
        want = est[(row["aggregation"], row["model"], row["metric"])]
        assert row["estimate"] == pytest.approx(want, abs=1e-14)


# ---------------------------------------------------------------------------
# Paired-record emission (e1_openet_paired_daily/v1) from the evaluator
# ---------------------------------------------------------------------------


def test_daily_volk_record_emitted_by_default(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=0)
    written = ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")
    record_path = tmp_path / ev.PAIRED_RECORD_FILENAME
    assert record_path.exists()
    assert written["paired_records"] == str(record_path)
    frame = ev.read_paired_record_frame(record_path)
    assert list(frame.columns) == list(ev.PAIRED_RECORD_COLUMNS)
    assert len(frame) == sum(r.n for r in two_site_cohort)
    # deterministically sorted by fid then date
    assert frame.equals(frame.sort_values(["fid", "date"], kind="mergesort"))


def test_record_roundtrip_recovers_exact_values(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=0)
    ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")
    frame = ev.read_paired_record_frame(tmp_path / ev.PAIRED_RECORD_FILENAME)
    rebuilt = {r.fid: r for r in ev.paired_records_from_frame(frame)}
    for rec in two_site_cohort:
        got = rebuilt[rec.fid]
        np.testing.assert_array_equal(got.observed, rec.observed)
        np.testing.assert_array_equal(got.swim, rec.swim)
        np.testing.assert_array_equal(got.openet, rec.openet)
        assert got.support_class == rec.support_class
        assert got.index.equals(rec.index)


def test_missing_support_metadata_blocks_canonical_daily_write(ev, tmp_path):
    bare = (
        _record(ev, "US-Aaa", 40, seed=31, support=False),
        _record(ev, "US-Bbb", 40, seed=32, support=False),
    )
    bundle = _bundle(ev, bare, reps=0)
    with pytest.raises(ev.GroupedEstimationError, match="support metadata"):
        ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")


def test_diy_and_monthly_never_write_canonical_record(ev, two_site_cohort, tmp_path):
    d_diy, d_mon = tmp_path / "diy", tmp_path / "monthly"
    d_diy.mkdir()
    d_mon.mkdir()
    ev.write_grouped_outputs(
        _bundle(ev, two_site_cohort, reps=0), str(d_diy), "daily", openet_source="diy"
    )
    monthly = _bundle(ev, two_site_cohort, scale="monthly", reps=0)
    ev.write_grouped_outputs(monthly, str(d_mon), "monthly", openet_source="volk")
    assert not (d_diy / ev.PAIRED_RECORD_FILENAME).exists()
    assert not (d_mon / ev.PAIRED_RECORD_FILENAME).exists()


def test_record_hash_and_contract_in_metadata(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=0)
    ev.write_grouped_outputs(bundle, str(tmp_path), "daily", openet_source="volk")
    with open(tmp_path / "evaluation_grouped_daily_metadata.json") as f:
        meta = json.load(f)
    contract = meta["paired_record_contract"]
    assert contract["schema_version"] == ev.PAIRED_RECORD_SCHEMA_VERSION
    assert contract["filename"] == ev.PAIRED_RECORD_FILENAME
    assert contract["sha256"] == meta["output_hashes"][ev.PAIRED_RECORD_FILENAME]
    assert contract["n_sites"] == 2
    assert contract["n_rows"] == sum(r.n for r in two_site_cohort)
    n_ret = sum(r.support_class.count("capture") for r in two_site_cohort)
    assert contract["n_retrieval"] == n_ret
    assert contract["n_between_retrieval"] == contract["n_rows"] - n_ret
    assert sum(contract["support_class_counts"].values()) == contract["n_rows"]
    assert contract["ordered_columns"] == list(ev.PAIRED_RECORD_COLUMNS)
    assert contract["minimum_all_days_count"] == ev.MIN_OBS_FOR_METRICS
    assert "unsupported" not in contract["allowed_support_classes"]


def test_roundtrip_gate_rejects_tampered_grouped_metrics(ev, two_site_cohort, tmp_path):
    bundle = _bundle(ev, two_site_cohort, reps=0)
    tampered = bundle.grouped_metrics.copy()
    tampered.loc[0, "estimate"] += 1e-6
    broken = ev.BenchmarkEvaluation(
        site_metrics=bundle.site_metrics,
        grouped_metrics=tampered,
        grouped_contrasts=bundle.grouped_contrasts,
        paired_records=bundle.paired_records,
        site_effect_summary=None,
        metadata=bundle.metadata,
    )
    with pytest.raises(ev.GroupedEstimationError, match="round-trip"):
        ev.write_grouped_outputs(broken, str(tmp_path), "daily", openet_source="volk")


def test_shared_module_reexports(ev):
    import swimrs.evaluation.benchmark as shared

    for name in (
        "PairedSiteSeries",
        "GroupedEstimationError",
        "grouped_point_estimates",
        "bootstrap_grouped",
        "bootstrap_grouped_from_counts",
        "paired_records_to_frame",
        "paired_records_from_frame",
        "read_paired_record_frame",
        "write_paired_record_frame",
        "validate_paired_record_frame",
        "site_secondary_metrics",
        "PAIRED_RECORD_SCHEMA_VERSION",
        "PAIRED_RECORD_FILENAME",
        "SUPPORT_CLASSES",
        "TEMPORAL_CLASSES",
    ):
        assert getattr(ev, name) is getattr(shared, name), name


# ---------------------------------------------------------------------------
# Default-output hygiene: no legacy aggregate, no NSE headline
# ---------------------------------------------------------------------------


def test_no_legacy_aggregate_strings_in_module(ev):
    source = SCRIPT.read_text()
    assert "PAIRED AGGREGATE" not in source
    assert "PAIRED MONTHLY AGGREGATE" not in source
    assert "win rate" not in source.lower()


def test_default_run_resolves_canonical_run22(ev, tmp_path):
    # Policy (examples/VALIDATION_POLICY.md): run22 is canonical; a bare run
    # must not silently resolve the superseded run21 configuration.
    for run in ("run21", "run22"):
        d = tmp_path / run
        d.mkdir()
        (d / "proj.3.par.csv").write_text("x")
    found = ev.find_reference_par_csv(str(tmp_path), "proj")
    assert found == str(tmp_path / "run22" / "proj.3.par.csv")

    source = SCRIPT.read_text()
    assert "_run21.swim" not in source
    assert "_run22.swim" in source


def test_grouped_summary_print_content(ev, two_site_cohort, capsys):
    bundle = _bundle(ev, two_site_cohort, reps=50)
    ev.print_grouped_summary(bundle, "daily")
    out = capsys.readouterr().out
    assert "GROUPED SWIM-OPENET BENCHMARK" in out
    assert "POOLED OBSERVATIONS" in out
    assert "SQRT(N)-WEIGHTED SITE METRICS" in out
    assert "NSE" not in out
    assert "win" not in out.lower()
    assert "SWIM-RS" in out and "OpenET ensemble" in out


# ---------------------------------------------------------------------------
# Caller-facing structure: wrappers and single forward-model run
# ---------------------------------------------------------------------------


def test_compatibility_wrappers_return_site_metrics(ev, monkeypatch):
    sentinel = pd.DataFrame({"n": [1]}, index=pd.Index(["US-X"], name="fid"))

    def fake_collect(*args, **kwargs):
        return sentinel, (), {}

    monkeypatch.setattr(ev, "_collect_daily", fake_collect)
    monkeypatch.setattr(ev, "_collect_monthly", fake_collect)
    out_d = ev.evaluate(None, None, "par.csv", [], "fluxdir", "volk")
    out_m = ev.evaluate_monthly(None, None, "par.csv", [], "fluxdir")
    assert out_d is sentinel
    assert out_m is sentinel


def test_bundle_functions_collect_once(ev, two_site_cohort, monkeypatch):
    calls = {"n": 0}
    site_metrics = pd.DataFrame(
        {"n": [r.n for r in two_site_cohort]},
        index=pd.Index([r.fid for r in two_site_cohort], name="fid"),
    )

    def fake_collect(*args, **kwargs):
        calls["n"] += 1
        return site_metrics, tuple(two_site_cohort), {}

    monkeypatch.setattr(ev, "_collect_daily", fake_collect)
    bundle = ev.evaluate_benchmark_daily(
        None, None, "par.csv", [], "fluxdir", bootstrap_reps=10, bootstrap_seed=42
    )
    assert calls["n"] == 1
    assert isinstance(bundle, ev.BenchmarkEvaluation)
    assert bundle.site_metrics is site_metrics
    assert len(bundle.grouped_metrics) == 18
    assert bundle.site_effect_summary is None
    b2 = ev.evaluate_benchmark_daily(
        None,
        None,
        "par.csv",
        [],
        "fluxdir",
        bootstrap_reps=10,
        bootstrap_seed=42,
        with_site_effect=True,
    )
    assert b2.site_effect_summary is not None
    assert calls["n"] == 2


def test_archive_run_uses_bundle_functions():
    src = (SCRIPT.parent / "archive_run.py").read_text()
    assert "evaluate_benchmark_daily" in src
    assert "evaluate_benchmark_monthly" in src
    assert "write_grouped_outputs" in src
    # the paired daily record is part of the canonical category-6 bundle;
    # its absence must be treated as an archive gap (no second evaluator call)
    assert "paired_records" in src
    assert "evaluation_paired_daily_records.csv" in src
    assert src.count("evaluate_benchmark_daily(") == 1


# ---------------------------------------------------------------------------
# Frozen-support regression: counts and grouped point targets
# ---------------------------------------------------------------------------

# Independent point checks derived from the corrected frozen daily/monthly
# support (45 sites / 59,516 site-days; 30 sites / 1,301 site-months). Run
# only when SWIM_E2_GROUPED_DIR points at a directory holding the grouped
# CSVs produced by evaluate.py on the canonical Run 22 footing.
GROUPED_DIR_ENV = "SWIM_E2_GROUPED_DIR"

DAILY_TARGETS = {
    ("pooled_observations", "swim", "kge"): 0.855837773,
    ("pooled_observations", "swim", "rmse"): 1.117361176,
    ("pooled_observations", "swim", "mbe"): -0.006843838,
    ("pooled_observations", "swim", "r"): 0.867429746,
    ("pooled_observations", "swim", "r2"): 0.752434364,
    ("pooled_observations", "swim", "slope0"): 0.923446511,
    ("pooled_observations", "openet_ensemble", "kge"): 0.820808357,
    ("pooled_observations", "openet_ensemble", "rmse"): 1.139236084,
    ("pooled_observations", "openet_ensemble", "mbe"): -0.211473920,
    ("pooled_observations", "openet_ensemble", "r"): 0.864912544,
    ("pooled_observations", "openet_ensemble", "r2"): 0.748073709,
    ("pooled_observations", "openet_ensemble", "slope0"): 0.866404414,
    ("sqrt_n_weighted_site_metric", "swim", "kge"): 0.782932081,
    ("sqrt_n_weighted_site_metric", "swim", "rmse"): 1.138474516,
    ("sqrt_n_weighted_site_metric", "swim", "mbe"): 0.026877340,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "kge"): 0.757754884,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "rmse"): 1.138840091,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "mbe"): -0.198191503,
}

MONTHLY_TARGETS = {
    ("pooled_observations", "swim", "kge"): 0.940662705,
    ("pooled_observations", "swim", "rmse"): 19.783547068,
    ("pooled_observations", "swim", "mbe"): 0.340454877,
    ("pooled_observations", "swim", "r"): 0.951460138,
    ("pooled_observations", "swim", "r2"): 0.905276395,
    ("pooled_observations", "swim", "slope0"): 0.973471191,
    ("pooled_observations", "openet_ensemble", "kge"): 0.912313360,
    ("pooled_observations", "openet_ensemble", "rmse"): 20.841344493,
    ("pooled_observations", "openet_ensemble", "mbe"): -3.821033541,
    ("pooled_observations", "openet_ensemble", "r"): 0.947709949,
    ("pooled_observations", "openet_ensemble", "r2"): 0.898154147,
    ("pooled_observations", "openet_ensemble", "slope0"): 0.934050373,
    ("sqrt_n_weighted_site_metric", "swim", "kge"): 0.847739457,
    ("sqrt_n_weighted_site_metric", "swim", "rmse"): 19.771203157,
    ("sqrt_n_weighted_site_metric", "swim", "mbe"): 1.463988761,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "kge"): 0.808232887,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "rmse"): 20.909893095,
    ("sqrt_n_weighted_site_metric", "openet_ensemble", "mbe"): -2.681482484,
}

FROZEN_SUPPORT = {"daily": (45, 59516), "monthly": (30, 1301)}


def _grouped_csv(scale):
    import os as _os

    base = _os.environ.get(GROUPED_DIR_ENV)
    if not base:
        pytest.skip(f"{GROUPED_DIR_ENV} not set (frozen-support regression is data-gated)")
    path = Path(base) / f"evaluation_grouped_{scale}_metrics.csv"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return pd.read_csv(path)


@pytest.mark.regression
@pytest.mark.parametrize("scale", ["daily", "monthly"])
def test_frozen_support_counts(scale):
    df = _grouped_csv(scale)
    n_sites, n_pairs = FROZEN_SUPPORT[scale]
    assert (df["n_sites"] == n_sites).all()
    assert (df["n_pairs"] == n_pairs).all()
    assert len(df) == 18


@pytest.mark.regression
@pytest.mark.parametrize("scale", ["daily", "monthly"])
def test_frozen_grouped_point_targets(scale):
    df = _grouped_csv(scale)
    targets = DAILY_TARGETS if scale == "daily" else MONTHLY_TARGETS
    got = df.set_index(["aggregation", "model", "metric"])["estimate"]
    for key, want in targets.items():
        assert got[key] == pytest.approx(want, abs=5e-9), key


# Canonical paired-record counts (plan §5.4): support-class rows must sum
# exactly to the all-days total on the Run 22 footing.
FROZEN_RECORD_COUNTS = {
    "n_sites": 45,
    "n_rows": 59516,
    "n_retrieval": 4987,
    "n_interpolated": 42048,
    "n_flat_fill": 12481,
    "n_between_retrieval": 54529,
}


@pytest.mark.regression
def test_frozen_paired_record_counts(ev):
    import os as _os

    base = _os.environ.get(GROUPED_DIR_ENV)
    if not base:
        pytest.skip(f"{GROUPED_DIR_ENV} not set (frozen-support regression is data-gated)")
    path = Path(base) / ev.PAIRED_RECORD_FILENAME
    if not path.exists():
        pytest.skip(f"{path} not found")
    frame = ev.read_paired_record_frame(path)
    counts = ev.validate_paired_record_frame(frame)
    assert counts["n_sites"] == FROZEN_RECORD_COUNTS["n_sites"]
    assert counts["n_rows"] == FROZEN_RECORD_COUNTS["n_rows"]
    assert counts["n_retrieval"] == FROZEN_RECORD_COUNTS["n_retrieval"]
    assert counts["n_between_retrieval"] == FROZEN_RECORD_COUNTS["n_between_retrieval"]
    sc = counts["support_class_counts"]
    assert sc["capture"] == FROZEN_RECORD_COUNTS["n_retrieval"]
    assert sc["interpolated"] == FROZEN_RECORD_COUNTS["n_interpolated"]
    assert sc["flat_fill"] == FROZEN_RECORD_COUNTS["n_flat_fill"]
    assert sum(sc.values()) == FROZEN_RECORD_COUNTS["n_rows"]
