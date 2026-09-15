"""Unit tests for the redefined E2 Gate G3 (daily ET consistency and low-ETo stability).

Covers the pure pieces of
``examples/6_Flux_International/e2_refooting/phase3_daily_basis_gate.py``: the ET-identity
read-back against written CSVs, the ingest-rule filter, the reproduction of the pest_builder
"spread" ensemble weights (two-member sample SD, spread floor, min-member rule), sum(w**2) shares,
and the threshold evaluation.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase3_daily_basis_gate", E2_DIR / "phase3_daily_basis_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


def _ledger(rows):
    cols = ["site", "date", "native", "eto", "etr", "ratio", "corrected", "status"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- consistency
def test_consistency_passes_on_identity_and_one_to_one(mod):
    ledger = _ledger(
        [
            ["S", "20150601", 0.5, 4.0, 5.0, 1.25, 0.625, "ok"],
            ["S", "20150617", 0.4, 2.0, 3.0, 1.5, 0.6, "ok"],
            ["S", "20150703", 0.3, np.nan, np.nan, np.nan, np.nan, "missing_sidecar"],
        ]
    )
    written = pd.DataFrame(
        {"site": ["S", "S"], "date": ["20150601", "20150617"], "written": [0.625, 0.6]}
    )
    r = mod.consistency_check(ledger, written, 1e-8)
    assert r["pass"]
    assert r["n_ok_ledger_rows"] == 2 and r["n_compared"] == 2
    assert r["n_non_ok_rows_written"] == 0 and r["max_rel_err"] <= 1e-12


def test_consistency_fails_on_missing_extra_nonok_or_value(mod):
    ledger = _ledger(
        [
            ["S", "20150601", 0.5, 4.0, 5.0, 1.25, 0.625, "ok"],
            ["S", "20150617", 0.4, 2.0, 3.0, 1.5, 0.6, "ok"],
            ["S", "20150703", 0.3, 1.0, 2.0, 2.0, 0.6, "nonpositive_eto"],
        ]
    )
    # ok row 20150617 missing; non-ok row 20150703 written; stray date 20150801 written
    written = pd.DataFrame(
        {
            "site": ["S", "S", "S"],
            "date": ["20150601", "20150703", "20150801"],
            "written": [0.625, 0.6, 0.1],
        }
    )
    r = mod.consistency_check(ledger, written, 1e-8)
    assert not r["pass"]
    assert r["n_ok_rows_missing_from_csvs"] == 1
    assert r["n_csv_values_without_ok_row"] == 2  # the non-ok date and the stray date
    assert r["n_non_ok_rows_written"] == 1

    # value drift beyond tolerance
    written_bad = pd.DataFrame(
        {"site": ["S", "S"], "date": ["20150601", "20150617"], "written": [0.625, 0.6001]}
    )
    r2 = mod.consistency_check(ledger, written_bad, 1e-8)
    assert not r2["pass"] and r2["max_rel_err"] > 1e-8


def test_read_grass_csvs_rejects_non_etf_columns_and_duplicates(mod, tmp_path):
    d = tmp_path / "no_mask"
    d.mkdir()
    pd.DataFrame({"sid": ["S"], "ETF_20150601": [0.6], "ETF_20150617": [0.5]}).to_csv(
        d / "ssebop_etf_grass_S_no_mask_2015.csv", index=False
    )
    out = mod.read_grass_csvs(str(d), {"S"}, (2013, 2025))
    assert list(out["date"]) == ["20150601", "20150617"]
    assert mod.read_grass_csvs(str(d), {"T"}, (2013, 2025)).empty
    pd.DataFrame({"sid": ["S"], "LE07_044033_20160601": [0.6]}).to_csv(
        d / "ssebop_etf_grass_S_no_mask_2016.csv", index=False
    )
    with pytest.raises(ValueError, match="unexpected column"):
        mod.read_grass_csvs(str(d), {"S"}, (2013, 2025))


# --------------------------------------------------------------------------- weights
def test_ingest_rules_match_ingestor_bounds(mod):
    v = pd.Series([0.04, 0.05, 1.0, 2.0, 2.01, np.nan])
    out = mod.apply_ingest_rules(v, 0.05, 2.0)
    assert out.isna().tolist() == [True, False, False, False, True, True]


def test_ensemble_weights_reproduce_pest_builder_spread_mode(mod):
    members = pd.DataFrame(
        {
            "ssebop": [0.6, 0.9, np.nan, 0.5],
            "ptjpl": [0.8, 0.9, 0.7, np.nan],
        }
    )
    w = mod.ensemble_weights(members, spread_floor=0.05, min_members=2, std_ddof=1)
    # two members: target = mean, SD = |a - b| / sqrt(2) (pandas ddof=1)
    assert w.loc[0, "target"] == pytest.approx(0.7)
    assert w.loc[0, "sd"] == pytest.approx(abs(0.6 - 0.8) / np.sqrt(2))
    assert w.loc[0, "weight"] == pytest.approx(0.7 / (0.2 / np.sqrt(2) + 0.05))
    # identical members: SD 0 -> weight = target / floor
    assert w.loc[1, "sd"] == 0.0
    assert w.loc[1, "weight"] == pytest.approx(0.9 / 0.05)
    # single member: obsval exists but weight is zero
    assert w.loc[2, "ct"] == 1 and not w.loc[2, "eligible"] and w.loc[2, "weight"] == 0.0
    assert w.loc[3, "target"] == pytest.approx(0.5) and w.loc[3, "weight"] == 0.0
    assert (w["w2"] == w["weight"] ** 2).all()
    # equals the pest_builder path literally: masked.std(axis=1) with the pandas default
    assert np.allclose(w["sd"].fillna(-1), members.std(axis=1).fillna(-1))
    assert not w["eto_floor_excluded"].any()

    # ETo floor (etf_weighting_eto_floor): inclusive threshold, zero weight below, target kept
    eto = pd.Series([0.6, 1.0, 5.0, 5.0])
    wf = mod.ensemble_weights(members, 0.05, 2, 1, eto=eto, eto_floor=1.0)
    assert wf.loc[0, "weight"] == 0.0 and wf.loc[0, "eto_floor_excluded"]
    assert wf.loc[0, "target"] == pytest.approx(0.7) and not wf.loc[0, "eligible"]
    assert wf.loc[1, "weight"] == pytest.approx(0.9 / 0.05)
    assert wf.loc[2, "weight"] == 0.0 and not wf.loc[2, "eto_floor_excluded"]
    with pytest.raises(ValueError, match="requires the daily eto"):
        mod.ensemble_weights(members, 0.05, 2, 1, eto_floor=1.0)
    with pytest.raises(ValueError, match="ETo missing"):
        mod.ensemble_weights(members, 0.05, 2, 1, eto=eto.where(eto > 1), eto_floor=1.0)


def test_w2_shares_sum_to_one_over_bins(mod):
    df = pd.DataFrame({"w2": [1.0, 3.0, 0.0, 4.0], "b": ["<1", "1-2", "<1", ">=3"]})
    s = mod.w2_shares(df, "w2", "b")
    assert s == {"<1": 0.125, "1-2": 0.375, ">=3": 0.5}
    assert mod.w2_shares(df.assign(w2=0.0), "w2", "b") == {}


# --------------------------------------------------------------------------- evaluation
def _members(rows):
    cols = ["site", "date", "ssebop_container", "ptjpl", "eto"]
    m = pd.DataFrame(rows, columns=cols)
    m["year"] = m["date"].str[:4].astype(int)
    return m


def test_assemble_and_evaluate_thresholds(mod):
    # site A: warm-season dates, well-behaved ratio; site B: one low-ETo winter date whose
    # corrected value is inflated and one corrected value above the ingest ceiling
    members = _members(
        [
            ["A", "20150601", 0.6, 0.7, 5.0],
            ["A", "20150617", 0.5, 0.6, 4.0],
            ["A", "20150703", np.nan, 0.8, 6.0],
            ["B", "20150115", 0.3, 0.2, 0.5],
            ["B", "20150601", 0.9, 0.8, 5.0],
            ["B", "20150617", 0.7, 0.75, 4.5],
        ]
    )
    ledger = _ledger(
        [
            ["A", "20150601", 0.6, 5.0, 6.0, 1.2, 0.72, "ok"],
            ["A", "20150617", 0.5, 4.0, 5.0, 1.25, 0.625, "ok"],
            ["B", "20150115", 0.3, 0.5, 3.0, 6.0, 1.8, "ok"],
            ["B", "20150601", 0.9, 5.0, 12.0, 2.4, 2.16, "ok"],
            ["B", "20150617", 0.7, 4.5, 5.4, 1.2, 0.84, "ok"],
        ]
    )
    t = mod.assemble(members, ledger)
    assert len(t) == 6
    a = t.set_index(["site", "date"])
    # corrected above 2.0 is dropped from the grass member and flagged
    assert np.isnan(a.loc[("B", "20150601"), "ssebop_grass"])
    assert a.loc[("B", "20150601"), "corrected_above_ceiling"]
    assert (
        a.loc[("B", "20150601"), "weight_grass"] == 0.0
        and a.loc[("B", "20150601"), "weight_native"] > 0
    )
    # single-member date carries zero weight in both configurations
    assert a.loc[("A", "20150703"), "weight_native"] == 0.0
    assert a.loc[("A", "20150703"), "weight_grass"] == 0.0
    # ratio screen flag and ETo bin
    assert a.loc[("B", "20150115"), "ratio_outside_screen"]
    assert str(a.loc[("B", "20150115"), "eto_bin"]) == "<1"
    assert str(a.loc[("A", "20150617"), "eto_bin"]) == ">=3"

    # the ETo floor (1.0 mm) zero-weights B/20150115 (ETo 0.5) in both configurations while the
    # target and the unfloored weights are kept for the diagnostic
    assert a.loc[("B", "20150115"), "eto_floor_excluded"]
    assert a.loc[("B", "20150115"), "weight_native"] == 0.0
    assert a.loc[("B", "20150115"), "weight_grass"] == 0.0
    assert a.loc[("B", "20150115"), "target_grass"] == pytest.approx((1.8 + 0.2) / 2)
    assert a.loc[("B", "20150115"), "weight_native_nofloor"] > 0
    # the inflated grass value disagrees with PT-JPL, so the spread weighting cuts its weight:
    # 1.0 / (1.6/sqrt(2) + 0.05) < 0.25 / (0.1/sqrt(2) + 0.05)
    assert (
        a.loc[("B", "20150115"), "weight_grass_nofloor"]
        < a.loc[("B", "20150115"), "weight_native_nofloor"]
    )
    assert (a["weight_native"] == a["weight_native_nofloor"])[~a["eto_floor_excluded"]].all()

    consistency = {"pass": True}
    r = mod.evaluate(t, consistency)
    g, s = r["gate"], r["summary"]
    assert g["daily_et_consistency"]
    # site B loses 1 of 3 paired dates to the ceiling -> per-site loss gate fails
    b = r["by_site"].set_index("site")
    assert b.loc["B", "ceiling_loss_frac_of_paired"] == pytest.approx(1 / 3)
    assert not g["ceiling_loss_ok"]
    # low-ETo share with the floor is zero by construction; the without-floor diagnostic
    # reports what the gate saw before the rule
    low = s["low_eto"]
    assert low["n_site_dates"] == 1
    assert low["n_weighted_grass"] == 0 and low["pooled_w2_share_grass"] == 0.0
    assert low["pooled_w2_share_increase"] == 0.0
    assert g["low_eto_pooled_share_ok"] and g["low_eto_pooled_increase_ok"]
    is_low = t["eto"] < 1.0
    nf = s["low_eto_without_floor"]
    assert nf["n_weighted_grass"] == 1
    assert nf["pooled_w2_share_grass"] == pytest.approx(
        t.loc[is_low, "w2_grass_nofloor"].sum() / t["w2_grass_nofloor"].sum()
    )
    assert nf["pooled_w2_share_native"] == pytest.approx(
        t.loc[is_low, "w2_native_nofloor"].sum() / t["w2_native_nofloor"].sum()
    )
    assert nf["pooled_w2_share_increase"] == pytest.approx(
        nf["pooled_w2_share_grass"] - nf["pooled_w2_share_native"]
    )
    assert nf["max_site"] == "B" and nf["max_site_w2_share_grass"] > 0
    ef = s["eto_floor"]
    assert ef["eto_floor_mm"] == 1.0
    assert ef["grass"]["obs_zeroed_by_floor"] == 1 and ef["native"]["obs_zeroed_by_floor"] == 1
    assert ef["grass"]["weighted_obs_with_floor"] == ef["grass"]["weighted_obs_without_floor"] - 1
    assert ef["grass"]["w2_share_removed"] == pytest.approx(nf["pooled_w2_share_grass"])
    assert sum(s["w2_share_by_eto_bin"]["grass"].values()) == pytest.approx(1.0)
    assert s["ingest_range"]["corrected_above_ceiling"] == 1
    assert not g["pass"]

    # a passing configuration: drop the ceiling case (the winter date is handled by the floor)
    t_ok = t[~((t["site"] == "B") & (t["date"] == "20150601"))]
    r_ok = mod.evaluate(t_ok.reset_index(drop=True), consistency)
    assert r_ok["gate"]["pass"]
    assert r_ok["summary"]["low_eto"]["n_site_dates"] == 1
    assert r_ok["summary"]["low_eto"]["n_weighted_grass"] == 0
    assert not r_ok["gate"]["ratio_screen_review_needed"]


def test_evaluate_flags_failed_consistency(mod):
    members = _members([["A", "20150601", 0.6, 0.7, 5.0]])
    ledger = _ledger([["A", "20150601", 0.6, 5.0, 6.0, 1.2, 0.72, "ok"]])
    t = mod.assemble(members, ledger)
    r = mod.evaluate(t, {"pass": False})
    assert not r["gate"]["daily_et_consistency"] and not r["gate"]["pass"]
