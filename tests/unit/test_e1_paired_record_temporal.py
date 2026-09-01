"""Unit tests for the e1_openet_paired_daily/v1 record contract and the
temporal decomposition estimators in ``swimrs.evaluation.benchmark``.

Covers:

- deterministic serialization, exact schema, and full-precision round-trip;
- row/cohort invariant enforcement (duplicates, non-finite ET, class enums,
  temporal/support consistency, sort order, thin sites);
- the common temporal cohort rule (>= 10 paired dates in BOTH classes) with
  explicit exclusion reasons and partition disjointness/union identities;
- pooled/weighted class estimates against independent direct formulas;
- the cross-model support interaction from one shared site-draw matrix
  (identical models give exactly zero; constructed relative degradation has
  the expected sign);
- signed-MBE preservation and default-output hygiene (no NSE/MAE/win-rate/
  abs-MBE in the grouped temporal outputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    MIN_OBS_FOR_METRICS,
    PAIRED_RECORD_COLUMNS,
    POOLED_METRICS,
    TEMPORAL_ALL_DAYS,
    TEMPORAL_CLASS_BETWEEN,
    TEMPORAL_CLASS_RETRIEVAL,
    WEIGHTED_METRICS,
    GroupedEstimationError,
    PairedSiteSeries,
    grouped_point_estimates,
    paired_records_from_frame,
    paired_records_to_frame,
    pooled_metrics_direct,
    read_paired_record_frame,
    site_metric_triads,
    temporal_class_records,
    temporal_cohort_from_frame,
    temporal_decomposition,
    validate_paired_record_frame,
    write_paired_record_frame,
)


def make_record(fid, n, seed, n_captures, swim_noise=0.3, openet_noise=0.3, between_extra=0.0):
    """Synthetic annotated site: captures at evenly spaced dates.

    ``between_extra`` adds extra SWIM noise on non-capture (between-retrieval)
    days only, to construct a relative degradation with a known sign.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    obs = np.abs(rng.normal(3.5, 1.2, size=n)) + 0.5
    support = np.array(["interpolated"] * n, dtype=object)
    cap_pos = np.linspace(0, n - 1, n_captures).round().astype(int)
    support[cap_pos] = "capture"
    if n > 4:
        for i in range(n - 2, n):
            if support[i] != "capture":
                support[i] = "flat_fill"
    swim = obs + rng.normal(0.0, swim_noise, size=n)
    if between_extra:
        between = support != "capture"
        swim[between] += rng.normal(0.0, between_extra, size=int(between.sum()))
    openet = obs + rng.normal(0.0, openet_noise, size=n) - 0.2
    return PairedSiteSeries(
        fid=fid, index=idx, observed=obs, swim=swim, openet=openet, support_class=tuple(support)
    )


@pytest.fixture(scope="module")
def cohort_records():
    """Sites spanning the eligibility space with unequal lengths."""
    return (
        make_record("S-Both", 120, seed=1, n_captures=15),
        make_record("S-FewRet", 120, seed=2, n_captures=5),  # < 10 retrieval
        make_record("S-FewBtw", 15, seed=3, n_captures=12),  # < 10 between
        make_record("S-Long", 400, seed=4, n_captures=40),
    )


@pytest.fixture(scope="module")
def frame(cohort_records):
    return paired_records_to_frame(cohort_records)


# ---------------------------------------------------------------------------
# Record contract: schema, ordering, round-trip
# ---------------------------------------------------------------------------


def test_frame_schema_and_deterministic_ordering(cohort_records):
    # feed records in reverse; the frame must come out sorted fid then date
    f1 = paired_records_to_frame(tuple(reversed(cohort_records)))
    f2 = paired_records_to_frame(cohort_records)
    assert f1.equals(f2)
    assert list(f1.columns) == list(PAIRED_RECORD_COLUMNS)
    assert f1.equals(f1.sort_values(["fid", "date"], kind="mergesort").reset_index(drop=True))


def test_full_precision_roundtrip(frame, tmp_path):
    path = tmp_path / "records.csv"
    write_paired_record_frame(frame, str(path))
    back = read_paired_record_frame(str(path))
    for col in ("flux_et_mm_d", "swim_et_mm_d", "openet_et_mm_d"):
        np.testing.assert_array_equal(back[col].to_numpy(), frame[col].to_numpy())
    assert back["fid"].tolist() == frame["fid"].tolist()
    assert (back["date"] == frame["date"]).all()
    assert back["openet_support_class"].tolist() == frame["openet_support_class"].tolist()
    assert back["temporal_class"].tolist() == frame["temporal_class"].tolist()


def test_roundtrip_grouped_identity(frame):
    est_orig = grouped_point_estimates(tuple(sorted_records := paired_records_from_frame(frame)))
    est_again = grouped_point_estimates(paired_records_from_frame(frame))
    for k in est_orig:
        assert est_orig[k] == pytest.approx(est_again[k], abs=1e-12)
    assert [r.fid for r in sorted_records] == sorted({r.fid for r in sorted_records})


def test_validate_counts_summary(frame, cohort_records):
    counts = validate_paired_record_frame(frame)
    assert counts["n_sites"] == 4
    assert counts["n_rows"] == sum(r.n for r in cohort_records)
    n_ret = sum(r.support_class.count("capture") for r in cohort_records)
    assert counts["n_retrieval"] == n_ret
    assert counts["n_between_retrieval"] == counts["n_rows"] - n_ret
    assert sum(counts["support_class_counts"].values()) == counts["n_rows"]


# ---------------------------------------------------------------------------
# Record contract: invariant rejections
# ---------------------------------------------------------------------------


def test_duplicate_fid_date_rejected(frame):
    bad = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    bad = bad.sort_values(["fid", "date"], kind="mergesort").reset_index(drop=True)
    with pytest.raises(GroupedEstimationError, match="duplicate"):
        validate_paired_record_frame(bad)


def test_nonfinite_et_rejected(frame):
    bad = frame.copy()
    bad.loc[5, "swim_et_mm_d"] = np.nan
    with pytest.raises(GroupedEstimationError, match="non-finite"):
        validate_paired_record_frame(bad)


def test_unsupported_class_rejected(frame):
    bad = frame.copy()
    bad.loc[3, "openet_support_class"] = "unsupported"
    with pytest.raises(GroupedEstimationError, match="unsupported"):
        validate_paired_record_frame(bad)


def test_unknown_class_rejected(frame):
    bad = frame.copy()
    bad.loc[3, "openet_support_class"] = "extrapolated"
    with pytest.raises(GroupedEstimationError, match="unknown support classes"):
        validate_paired_record_frame(bad)


def test_temporal_support_inconsistency_rejected(frame):
    bad = frame.copy()
    # flip one temporal label without touching the support class
    i = bad.index[bad["openet_support_class"] == "interpolated"][0]
    bad.loc[i, "temporal_class"] = TEMPORAL_CLASS_RETRIEVAL
    with pytest.raises(GroupedEstimationError, match="inconsistent"):
        validate_paired_record_frame(bad)


def test_unknown_temporal_class_rejected(frame):
    bad = frame.copy()
    bad.loc[0, "temporal_class"] = "overpass"
    with pytest.raises(GroupedEstimationError, match="unknown temporal classes"):
        validate_paired_record_frame(bad)


def test_unsorted_rows_rejected(frame):
    bad = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(GroupedEstimationError, match="sorted"):
        validate_paired_record_frame(bad)


def test_extra_column_rejected(frame):
    bad = frame.copy()
    bad["is_overpass"] = False
    with pytest.raises(GroupedEstimationError, match="schema violation"):
        validate_paired_record_frame(bad)


def test_thin_site_rejected(frame):
    thin = make_record("S-Both", 120, seed=1, n_captures=15)
    keep = frame[frame["fid"] != "S-FewBtw"]
    stub = paired_records_to_frame((thin,)).iloc[: MIN_OBS_FOR_METRICS - 1]
    stub = stub.assign(fid="A-Thin")
    bad = (
        pd.concat([keep, stub], ignore_index=True)
        .sort_values(["fid", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    with pytest.raises(GroupedEstimationError, match="MIN_OBS_FOR_METRICS"):
        validate_paired_record_frame(bad)


def test_missing_support_metadata_rejected():
    rec = make_record("S-Both", 40, seed=9, n_captures=6)
    bare = PairedSiteSeries(
        fid=rec.fid, index=rec.index, observed=rec.observed, swim=rec.swim, openet=rec.openet
    )
    with pytest.raises(GroupedEstimationError, match="support metadata"):
        paired_records_to_frame((bare,))


def test_unsupported_class_rejected_at_series_level():
    rec = make_record("S-Both", 40, seed=9, n_captures=6)
    support = list(rec.support_class)
    support[4] = "unsupported"
    with pytest.raises(GroupedEstimationError, match="prohibited"):
        PairedSiteSeries(
            fid=rec.fid,
            index=rec.index,
            observed=rec.observed,
            swim=rec.swim,
            openet=rec.openet,
            support_class=tuple(support),
        )


# ---------------------------------------------------------------------------
# Temporal cohort
# ---------------------------------------------------------------------------


def test_common_cohort_membership_and_reasons(frame):
    eligibility, common = temporal_cohort_from_frame(frame)
    assert common == ("S-Both", "S-Long")
    elig = eligibility.set_index("fid")
    assert not elig.loc["S-FewRet", "eligible_retrieval"]
    assert "n_retrieval=5 < 10" in elig.loc["S-FewRet", "exclusion_reason"]
    assert not elig.loc["S-FewBtw", "eligible_between_retrieval"]
    assert "n_between_retrieval=3 < 10" in elig.loc["S-FewBtw", "exclusion_reason"]
    assert elig.loc["S-Both", "in_common_cohort"]
    assert elig.loc["S-Both", "exclusion_reason"] == ""
    # class-specific n values are exact
    assert elig.loc["S-Both", "n_retrieval"] == 15
    assert elig.loc["S-Both", "n_all_days"] == 120
    assert (elig["n_retrieval"] + elig["n_between_retrieval"] == elig["n_all_days"]).all()
    assert (elig["n_interpolated"] + elig["n_flat_fill"] == elig["n_between_retrieval"]).all()


def test_partitions_disjoint_union_ordered(frame):
    _, common = temporal_cohort_from_frame(frame)
    parts = temporal_class_records(frame, common)
    fids = tuple(sorted(common))
    for name in (TEMPORAL_ALL_DAYS, TEMPORAL_CLASS_RETRIEVAL, TEMPORAL_CLASS_BETWEEN):
        assert tuple(r.fid for r in parts[name]) == fids
    for a, r, b in zip(
        parts[TEMPORAL_ALL_DAYS],
        parts[TEMPORAL_CLASS_RETRIEVAL],
        parts[TEMPORAL_CLASS_BETWEEN],
        strict=True,
    ):
        assert len(r.index.intersection(b.index)) == 0
        assert r.index.union(b.index).equals(a.index)
        assert r.n + b.n == a.n
        # both models are scored on identical dates inside each class
        assert len(r.observed) == len(r.swim) == len(r.openet)


def test_no_common_cohort_raises():
    rec = make_record("S-FewRet", 60, seed=8, n_captures=4)
    frame = paired_records_to_frame((rec,))
    with pytest.raises(GroupedEstimationError, match="no common temporal cohort"):
        temporal_cohort_from_frame(frame)


def test_missing_subset_site_raises(frame):
    with pytest.raises(GroupedEstimationError, match="missing from paired record"):
        paired_records_from_frame(
            frame, fids=["S-Both", "S-Nope"], temporal_class=TEMPORAL_CLASS_RETRIEVAL
        )


# ---------------------------------------------------------------------------
# Temporal estimators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def decomp(frame):
    return temporal_decomposition(frame, reps=200, seed=42)


def test_pooled_class_metrics_match_direct_concatenation(frame, decomp):
    _, common = temporal_cohort_from_frame(frame)
    parts = temporal_class_records(frame, common)
    g = decomp.grouped_metrics.set_index(["temporal_class", "aggregation", "model", "metric"])[
        "estimate"
    ]
    for name, records in parts.items():
        for model, attr in (("swim", "swim"), ("openet_ensemble", "openet")):
            obs = np.concatenate([r.observed for r in records])
            mod = np.concatenate([getattr(r, attr) for r in records])
            direct = pooled_metrics_direct(obs, mod)
            for k in POOLED_METRICS:
                assert g[(name, AGG_POOLED, model, k)] == pytest.approx(direct[k], abs=1e-12), (
                    name,
                    model,
                    k,
                )


def test_weighted_class_metrics_match_independent_formula(frame, decomp):
    _, common = temporal_cohort_from_frame(frame)
    parts = temporal_class_records(frame, common)
    g = decomp.grouped_metrics.set_index(["temporal_class", "aggregation", "model", "metric"])[
        "estimate"
    ]
    for name, records in parts.items():
        w = np.sqrt(np.array([r.n for r in records], dtype=float))
        for model in ("swim", "openet_ensemble"):
            triads = site_metric_triads(records, model)
            for k in WEIGHTED_METRICS:
                expected = np.sum(w * triads[k]) / np.sum(w)
                assert g[(name, AGG_WEIGHTED, model, k)] == pytest.approx(expected, abs=1e-12), (
                    name,
                    model,
                    k,
                )


def test_contrasts_equal_model_differences(decomp):
    g = decomp.grouped_metrics.set_index(["temporal_class", "aggregation", "model", "metric"])[
        "estimate"
    ]
    for _, row in decomp.grouped_contrasts.iterrows():
        key = (row["temporal_class"], row["aggregation"])
        expected = g[(*key, "swim", row["metric"])] - g[(*key, "openet_ensemble", row["metric"])]
        assert row["estimate"] == pytest.approx(expected, abs=1e-14)


def test_interaction_equals_difference_of_class_contrasts(decomp):
    c = decomp.grouped_contrasts.set_index(["temporal_class", "aggregation", "metric"])["estimate"]
    for _, row in decomp.interactions.iterrows():
        agg, k = row["aggregation"], row["metric"]
        expected = c[(TEMPORAL_CLASS_BETWEEN, agg, k)] - c[(TEMPORAL_CLASS_RETRIEVAL, agg, k)]
        assert row["estimate"] == pytest.approx(expected, abs=1e-14)
        assert row["interaction"] == "between_retrieval_minus_retrieval_of_swim_minus_openet"


def test_identical_models_zero_contrasts_and_interactions():
    # swim == openet everywhere: shared site draws force every contrast and
    # interaction replicate to exactly zero — point estimates AND CI bounds
    records = []
    for i, (n, caps) in enumerate([(80, 12), (200, 25)]):
        rec = make_record(f"S-{i}", n, seed=50 + i, n_captures=caps)
        records.append(
            PairedSiteSeries(
                fid=rec.fid,
                index=rec.index,
                observed=rec.observed,
                swim=rec.swim,
                openet=rec.swim.copy(),
                support_class=rec.support_class,
            )
        )
    frame = paired_records_to_frame(tuple(records))
    d = temporal_decomposition(frame, reps=100, seed=42)
    for col in ("estimate", "ci95_low", "ci95_high"):
        np.testing.assert_allclose(d.grouped_contrasts[col].to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(d.interactions[col].to_numpy(), 0.0, atol=1e-12)


def test_constructed_degradation_interaction_sign():
    # SWIM degrades between retrievals (extra noise on non-capture days) while
    # OpenET does not: the pooled RMSE interaction must be positive (relative
    # SWIM loss between retrievals)
    records = tuple(
        make_record(f"S-{i}", 300, seed=60 + i, n_captures=30, between_extra=1.5) for i in range(3)
    )
    frame = paired_records_to_frame(records)
    d = temporal_decomposition(frame, reps=0, seed=42)
    inter = d.interactions.set_index(["aggregation", "metric"])["estimate"]
    assert inter[(AGG_POOLED, "rmse")] > 0.1
    assert inter[(AGG_WEIGHTED, "rmse")] > 0.1


def test_signed_mbe_stays_signed(frame, decomp):
    # the synthetic OpenET runs 0.2 mm/d low: its MBE must be negative in
    # every temporal class, never folded to a magnitude
    g = decomp.grouped_metrics
    mbe = g[(g["model"] == "openet_ensemble") & (g["metric"] == "mbe")]
    assert (mbe["estimate"] < 0).all()


def test_default_outputs_exclude_legacy_metrics(decomp):
    for df in (decomp.grouped_metrics, decomp.grouped_contrasts, decomp.interactions):
        metrics = set(df["metric"].unique())
        assert metrics <= set(POOLED_METRICS)
        for banned in ("nse", "mae", "abs_mbe", "win_rate"):
            assert banned not in metrics
        assert not any("abs_mbe" in c or "win" in c for c in df.columns)


def test_zero_reps_null_ci_only_when_explicit(frame, decomp):
    d0 = temporal_decomposition(frame, reps=0, seed=42)
    assert d0.grouped_metrics["ci95_low"].isna().all()
    assert d0.interactions["ci95_low"].isna().all()
    assert decomp.grouped_metrics["ci95_low"].notna().all()
    assert decomp.interactions["ci95_low"].notna().all()


def test_negative_reps_rejected(frame):
    with pytest.raises(GroupedEstimationError, match="non-negative integer"):
        temporal_decomposition(frame, reps=-1, seed=42)


def test_seed_determinism():
    # enough eligible sites that different seeds cannot coincide on every CI
    records = tuple(
        make_record(f"S-{i}", 90 + 15 * i, seed=70 + i, n_captures=12 + i) for i in range(6)
    )
    rich = paired_records_to_frame(records)
    d1 = temporal_decomposition(rich, reps=100, seed=42)
    d2 = temporal_decomposition(rich, reps=100, seed=42)
    d3 = temporal_decomposition(rich, reps=100, seed=7)
    assert d1.grouped_metrics.equals(d2.grouped_metrics)
    assert d1.interactions.equals(d2.interactions)
    assert not d1.grouped_metrics["ci95_low"].equals(d3.grouped_metrics["ci95_low"])


def test_favorable_direction_metadata(decomp):
    c = decomp.grouped_contrasts.set_index(["temporal_class", "aggregation", "metric"])
    i = decomp.interactions.set_index(["aggregation", "metric"])
    for agg in (AGG_POOLED, AGG_WEIGHTED):
        for tc in (TEMPORAL_ALL_DAYS, TEMPORAL_CLASS_RETRIEVAL, TEMPORAL_CLASS_BETWEEN):
            assert c.loc[(tc, agg, "kge"), "favorable_direction"] == "positive"
            assert c.loc[(tc, agg, "rmse"), "favorable_direction"] == "negative"
            assert c.loc[(tc, agg, "mbe"), "favorable_direction"] == "directional_only"
        assert i.loc[(agg, "kge"), "favorable_direction"] == "positive"
        assert i.loc[(agg, "rmse"), "favorable_direction"] == "negative"
        assert i.loc[(agg, "mbe"), "favorable_direction"] == "directional_only"


def test_class_counts_and_cohort(frame, decomp):
    eligibility, common = temporal_cohort_from_frame(frame)
    assert decomp.common_cohort == common
    elig = eligibility.set_index("fid").loc[list(common)]
    assert decomp.class_counts[TEMPORAL_ALL_DAYS] == int(elig["n_all_days"].sum())
    assert decomp.class_counts[TEMPORAL_CLASS_RETRIEVAL] == int(elig["n_retrieval"].sum())
    assert decomp.class_counts[TEMPORAL_CLASS_BETWEEN] == int(elig["n_between_retrieval"].sum())
