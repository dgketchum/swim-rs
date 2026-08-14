"""Unit tests for the E2 conditioned-ensemble uncertainty diagnostic (Example 5)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"
SCRIPT = EXAMPLE_DIR / "conditioned_ensemble_uncertainty.py"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
SPEC = importlib.util.spec_from_file_location("conditioned_ensemble_uncertainty", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _realization_frame(ids, n_cols=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.5, 0.1, size=(len(ids), n_cols)),
        index=pd.Index([str(i) for i in ids], name="real_name"),
        columns=[f"obs{j}" for j in range(n_cols)],
    )


def _cohort_and_meta():
    site = ["S1", "S1", "S2", "S2"]
    date = ["2020-05-01", "2020-06-02", "2020-05-01", "2020-07-03"]
    target = [0.40, 0.55, 0.61, 0.32]
    spread = [0.05, 0.09, 0.02, 0.11]
    cohort = pd.DataFrame(
        {
            "site": site,
            "date": date,
            "target": target,
            "spread": spread,
            "flux_etf": [0.44, 0.51, 0.58, 0.36],
        }
    )
    etf_meta = pd.DataFrame(
        {
            "obsnme": [f"oname:etf_{s}_{d}".replace("-", "") for s, d in zip(site, date)],
            "site": site,
            "date": date,
            "target_etf": target,
            "ensemble_std": spread,
        }
    )
    return cohort, etf_meta


def _synthetic_obs(row_counts, seed=3):
    """Capture-level observation table with the columns the analyses consume."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, n in enumerate(row_counts):
        spread = rng.uniform(0.01, 0.25, n)
        iqr = rng.uniform(0.01, 0.20, n)
        err = np.abs(0.35 * spread + 0.15 * iqr + rng.normal(0.0, 0.04, n))
        frames.append(
            pd.DataFrame(
                {
                    "site": f"S{i}",
                    "spread_retrieval": spread,
                    "spread_conditioned_iqr": iqr,
                    "spread_conditioned_std": iqr / 1.35,
                    "width90": iqr * 2.4,
                    "abs_error_effective": err,
                    "abs_error_ensemble_median": err * 0.9 + 0.005,
                    "covered_90": rng.random(n) < 0.7,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _ci(point, lo, hi):
    return {"point": point, "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# 1. Exclusion of the base realization (Gate C)
# ---------------------------------------------------------------------------


def test_numbered_realizations_exclude_base_and_sort_numerically():
    df = _realization_frame(["base", "10", "2", "1"])
    out = MODULE.select_numbered_realizations(df, expected=3)

    assert list(out.index) == ["1", "2", "10"]
    assert "base" not in out.index
    pd.testing.assert_series_equal(out.loc["10"], df.loc["10"])


def test_missing_base_realization_raises():
    df = _realization_frame(["0", "1", "2"])
    with pytest.raises(MODULE.GateError, match="base realization not present"):
        MODULE.select_numbered_realizations(df, expected=3)


def test_non_numeric_realization_id_raises():
    df = _realization_frame(["base", "0", "mean"])
    with pytest.raises(MODULE.GateError, match="non-numeric realization ids"):
        MODULE.select_numbered_realizations(df, expected=2)


def test_unexpected_numbered_count_raises():
    df = _realization_frame(["base", "0", "1"])
    with pytest.raises(MODULE.GateError, match="expected 3"):
        MODULE.select_numbered_realizations(df, expected=3)


def test_expected_none_skips_count_check():
    df = _realization_frame(["base", "0", "1"])
    out = MODULE.select_numbered_realizations(df, expected=None)

    assert list(out.index) == ["0", "1"]


# ---------------------------------------------------------------------------
# 2. One-to-one site-date -> obsnme mapping (Gate B)
# ---------------------------------------------------------------------------


def test_cohort_maps_one_to_one_onto_obsnme():
    cohort, etf_meta = _cohort_and_meta()
    merged = MODULE.map_cohort_to_obsnme(cohort, etf_meta)

    assert len(merged) == len(cohort)
    assert list(merged["obsnme"]) == list(etf_meta["obsnme"])
    assert "target_etf" not in merged.columns
    assert "ensemble_std" not in merged.columns
    assert list(merged["spread"]) == list(cohort["spread"])


def test_target_agreement_within_tolerance_is_accepted():
    cohort, etf_meta = _cohort_and_meta()
    etf_meta.loc[0, "target_etf"] += 1e-12
    etf_meta.loc[1, "ensemble_std"] -= 1e-12

    merged = MODULE.map_cohort_to_obsnme(cohort, etf_meta)
    assert len(merged) == len(cohort)


def test_duplicated_cohort_keys_raise():
    cohort, etf_meta = _cohort_and_meta()
    cohort = pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True)

    with pytest.raises(MODULE.GateError, match="duplicated site-date keys in frozen cohort"):
        MODULE.map_cohort_to_obsnme(cohort, etf_meta)


def test_duplicated_metadata_keys_raise():
    cohort, etf_meta = _cohort_and_meta()
    dup = etf_meta.iloc[[0]].copy()
    dup["obsnme"] = "oname:etf_duplicate"
    etf_meta = pd.concat([etf_meta, dup], ignore_index=True)

    with pytest.raises(MODULE.GateError, match="duplicated site-date keys in ETf metadata"):
        MODULE.map_cohort_to_obsnme(cohort, etf_meta)


def test_unmapped_capture_raises():
    cohort, etf_meta = _cohort_and_meta()
    etf_meta = etf_meta.drop(index=2).reset_index(drop=True)

    with pytest.raises(MODULE.GateError, match="1 captures unmapped"):
        MODULE.map_cohort_to_obsnme(cohort, etf_meta)


def test_archived_target_disagreement_raises():
    cohort, etf_meta = _cohort_and_meta()
    etf_meta.loc[2, "target_etf"] += 1e-6

    with pytest.raises(MODULE.GateError, match="archived target/spread disagree"):
        MODULE.map_cohort_to_obsnme(cohort, etf_meta)


def test_archived_spread_disagreement_raises():
    cohort, etf_meta = _cohort_and_meta()
    etf_meta.loc[3, "ensemble_std"] -= 1e-6

    with pytest.raises(MODULE.GateError, match="archived target/spread disagree"):
        MODULE.map_cohort_to_obsnme(cohort, etf_meta)


# ---------------------------------------------------------------------------
# 3. Conditioned quantiles on a small synthetic ensemble
# ---------------------------------------------------------------------------


def test_conditioned_stats_match_hand_computed_quantiles():
    rng = np.random.default_rng(7)
    arr = rng.normal(0.5, 0.1, size=(9, 4))
    matrix = pd.DataFrame(arr, index=[str(i) for i in range(9)], columns=["c0", "c1", "c2", "c3"])

    out = MODULE.conditioned_stats(matrix)
    q05, q25, q50, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)

    assert list(out.index) == ["c0", "c1", "c2", "c3"]
    np.testing.assert_allclose(out["q05"].to_numpy(), q05)
    np.testing.assert_allclose(out["q25"].to_numpy(), q25)
    np.testing.assert_allclose(out["q50"].to_numpy(), q50)
    np.testing.assert_allclose(out["q75"].to_numpy(), q75)
    np.testing.assert_allclose(out["q95"].to_numpy(), q95)
    np.testing.assert_allclose(out["spread_conditioned_std"].to_numpy(), arr.std(axis=0, ddof=1))
    np.testing.assert_allclose(out["spread_conditioned_iqr"].to_numpy(), q75 - q25)
    np.testing.assert_allclose(out["width90"].to_numpy(), q95 - q05)


def test_conditioned_stats_reject_non_finite_values():
    matrix = pd.DataFrame(
        [[0.4, 0.5], [0.6, np.nan], [0.5, 0.55]], index=["0", "1", "2"], columns=["a", "b"]
    )

    with pytest.raises(MODULE.GateError, match="non-finite values"):
        MODULE.conditioned_stats(matrix)


# ---------------------------------------------------------------------------
# 4. Paired whole-site bootstrap behavior
# ---------------------------------------------------------------------------


def test_bootstrap_site_median_point_is_the_median():
    out = MODULE.bootstrap_site_median([0.1, 0.2, 0.3, 0.4], n_boot=200, seed=42)

    assert out["point"] == pytest.approx(0.25)
    assert out["n_sites"] == 4
    assert out["n_boot"] == 200
    assert out["seed"] == 42
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]


def test_bootstrap_site_median_single_site_is_degenerate():
    out = MODULE.bootstrap_site_median([0.37], n_boot=100, seed=42)

    assert out["point"] == pytest.approx(0.37)
    assert out["ci_lo"] == pytest.approx(0.37)
    assert out["ci_hi"] == pytest.approx(0.37)
    assert out["n_sites"] == 1


def test_bootstrap_site_median_rejects_empty_and_non_finite():
    with pytest.raises(ValueError, match="non-empty and finite"):
        MODULE.bootstrap_site_median([], n_boot=10, seed=42)
    with pytest.raises(ValueError, match="non-empty and finite"):
        MODULE.bootstrap_site_median([0.1, np.nan, 0.3], n_boot=10, seed=42)


def test_pooled_bootstrap_single_site_gives_degenerate_intervals():
    obs = _synthetic_obs([14], seed=11)
    out = MODULE.pooled_site_bootstrap(obs, n_boot=25, seed=42)

    for key in [
        "pooled_rho_retrieval",
        "pooled_rho_conditioned_iqr",
        "pooled_rho_difference",
        "pooled_coverage_90",
    ]:
        stat = out[key]
        assert np.isfinite(stat["point"])
        assert stat["ci_lo"] == pytest.approx(stat["point"])
        assert stat["ci_hi"] == pytest.approx(stat["point"])

    assert out["pooled_coverage_90"]["point"] == pytest.approx(obs["covered_90"].mean())
    assert out["pooled_rho_difference"]["point"] == pytest.approx(
        out["pooled_rho_retrieval"]["point"] - out["pooled_rho_conditioned_iqr"]["point"]
    )
    assert np.isfinite(out["pooled_rho_retrieval_vs_conditioned"])


# ---------------------------------------------------------------------------
# 5. Deterministic output under a fixed seed
# ---------------------------------------------------------------------------


def test_bootstrap_site_median_is_seed_deterministic():
    rng = np.random.default_rng(17)
    values = rng.normal(0.1, 0.3, 60)

    a = MODULE.bootstrap_site_median(values, n_boot=500, seed=42)
    b = MODULE.bootstrap_site_median(values, n_boot=500, seed=42)
    c = MODULE.bootstrap_site_median(values, n_boot=500, seed=7)

    assert a == b
    assert c["point"] == pytest.approx(a["point"])
    assert (c["ci_lo"], c["ci_hi"]) != (a["ci_lo"], a["ci_hi"])


def test_pooled_site_bootstrap_is_seed_deterministic():
    obs = _synthetic_obs([30] * 8, seed=5)

    a = MODULE.pooled_site_bootstrap(obs, n_boot=100, seed=42)
    b = MODULE.pooled_site_bootstrap(obs, n_boot=100, seed=42)
    c = MODULE.pooled_site_bootstrap(obs, n_boot=100, seed=7)

    assert a == b
    assert c["pooled_rho_retrieval"]["point"] == pytest.approx(a["pooled_rho_retrieval"]["point"])
    assert (
        c["pooled_rho_retrieval"]["ci_lo"],
        c["pooled_rho_retrieval"]["ci_hi"],
    ) != (a["pooled_rho_retrieval"]["ci_lo"], a["pooled_rho_retrieval"]["ci_hi"])


# ---------------------------------------------------------------------------
# 6. Undefined correlations are recorded, never silently dropped
# ---------------------------------------------------------------------------


def _persite_obs():
    rows = []
    for site, n, spread in [("CONST", 8, None), ("SMALL", 3, None), ("VARY", 8, "varying")]:
        for i in range(n):
            rows.append(
                {
                    "site": site,
                    "spread_retrieval": 0.01 + 0.02 * i if spread else 0.05,
                    "spread_conditioned_iqr": 0.02 + 0.01 * i,
                    "spread_conditioned_std": 0.015 + 0.008 * i,
                    "width90": 0.05 + 0.02 * i,
                    "abs_error_effective": 0.03 + 0.02 * ((i * 3) % 7),
                    "abs_error_ensemble_median": 0.04 + 0.015 * ((i * 5) % 7),
                    "covered_90": i % 2 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_constant_spread_site_is_kept_and_its_undefined_rho_recorded():
    obs = _persite_obs()
    persite, undefined = MODULE.persite_associations(obs, min_obs=5)
    rows = persite.set_index("site")

    assert list(persite["site"]) == ["CONST", "SMALL", "VARY"]
    assert np.isnan(rows.loc["CONST", "rho_retrieval"])
    assert bool(rows.loc["CONST", "eligible"])
    assert rows.loc["CONST", "rho_retrieval_undefined_cause"] == "constant_spread"
    assert np.isnan(rows.loc["CONST", "delta_rho"])
    assert {"site": "CONST", "measure": "retrieval", "cause": "constant_spread"} in undefined
    assert np.isfinite(rows.loc["VARY", "rho_retrieval"])
    assert not any(u["site"] == "VARY" for u in undefined)


def test_ineligible_small_site_is_kept_but_not_listed_as_undefined():
    obs = _persite_obs()
    persite, undefined = MODULE.persite_associations(obs, min_obs=5)
    rows = persite.set_index("site")

    assert "SMALL" in rows.index
    assert not bool(rows.loc["SMALL", "eligible"])
    assert rows.loc["SMALL", "n"] == 3
    assert np.isnan(rows.loc["SMALL", "rho_retrieval"])
    assert not any(u["site"] == "SMALL" for u in undefined)


# ---------------------------------------------------------------------------
# 7. Quintile diagnostics
# ---------------------------------------------------------------------------


def test_quintile_diagnostics_cover_both_analyses_and_measures():
    obs = _synthetic_obs([26, 29, 23, 31], seed=9)
    table = MODULE.quintile_diagnostics(obs)

    assert set(table["analysis"]) == {"raw", "within_site_rank"}
    assert set(table["spread_measure"]) == {"retrieval", "conditioned_iqr"}
    for analysis in ["raw", "within_site_rank"]:
        for measure in ["retrieval", "conditioned_iqr"]:
            sub = table[(table["analysis"] == analysis) & (table["spread_measure"] == measure)]
            assert list(sub["quintile"]) == [1, 2, 3, 4, 5]
            assert sub["n"].sum() == len(obs)


def test_within_site_rank_quintiles_are_invariant_to_monotone_rescaling():
    obs = _synthetic_obs([26, 29, 23, 31], seed=9)
    rescaled = obs.copy()
    mask = rescaled["site"] == "S0"
    rescaled.loc[mask, "spread_retrieval"] *= 10.0

    def _rank_table(frame):
        t = MODULE.quintile_diagnostics(frame)
        t = t[(t["analysis"] == "within_site_rank") & (t["spread_measure"] == "retrieval")]
        return t[["quintile", "n", "MAE", "RMSE"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(_rank_table(obs), _rank_table(rescaled))


# ---------------------------------------------------------------------------
# 8. Mechanical outcome classification (plan section 10)
# ---------------------------------------------------------------------------


def test_delta_ci_above_zero_gives_outcome_a():
    out = MODULE.classify_outcome(_ci(0.12, 0.03, 0.21), _ci(0.05, -0.02, 0.13))

    assert out["comparison_outcome"] == "A"
    assert out["primary_outcome"] == "A"
    assert out["conditioned_informative"] is False
    assert out["outcome_d_applies"] is True


def test_spanning_delta_with_informative_conditioned_gives_outcome_c():
    out = MODULE.classify_outcome(_ci(0.02, -0.05, 0.09), _ci(0.11, 0.04, 0.19))

    assert out["comparison_outcome"] == "B"
    assert out["primary_outcome"] == "C"
    assert out["conditioned_informative"] is True
    assert out["outcome_d_applies"] is False


def test_spanning_delta_and_uninformative_conditioned_gives_outcome_b():
    out = MODULE.classify_outcome(_ci(0.01, -0.06, 0.08), _ci(0.02, -0.04, 0.09))

    assert out["primary_outcome"] == "B"
    assert out["outcome_d_applies"] is True


def test_delta_below_zero_without_informative_conditioned_gives_outcome_d():
    out = MODULE.classify_outcome(_ci(-0.11, -0.20, -0.03), _ci(0.01, -0.05, 0.07))

    assert out["comparison_outcome"] == "conditioned_favored"
    assert out["primary_outcome"] == "D"
    assert out["outcome_d_applies"] is True


def test_delta_below_zero_with_informative_conditioned_gives_outcome_c():
    out = MODULE.classify_outcome(_ci(-0.11, -0.20, -0.03), _ci(0.11, 0.04, 0.19))

    assert out["comparison_outcome"] == "conditioned_favored"
    assert out["primary_outcome"] == "C"
    assert out["outcome_d_applies"] is False


def test_outcome_d_is_determined_solely_by_conditioned_interval():
    uninformative = MODULE.classify_outcome(_ci(0.01, -0.06, 0.08), _ci(0.02, -0.04, 0.09))
    assert uninformative["outcome_d_applies"] is True

    boundary = MODULE.classify_outcome(_ci(0.01, -0.06, 0.08), _ci(0.05, 0.0, 0.11))
    assert boundary["conditioned_informative"] is False
    assert boundary["outcome_d_applies"] is True

    informative = MODULE.classify_outcome(_ci(0.01, -0.06, 0.08), _ci(0.11, 0.04, 0.19))
    assert informative["primary_outcome"] == "C"
    assert informative["outcome_d_applies"] is False


def test_outcome_d_can_accompany_outcome_a():
    out = MODULE.classify_outcome(_ci(0.14, 0.05, 0.23), _ci(0.01, -0.05, 0.07))

    assert out["primary_outcome"] == "A"
    assert out["outcome_d_applies"] is True
