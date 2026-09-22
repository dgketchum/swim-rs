"""Unit tests for the E2 inverse-problem audit and pre-launch archive helpers.

Covers the pure pieces of ``objective_audit.py`` (observation-name decoding,
independent reconstruction of the pest_builder spread weights with member-count and ETo-floor
rules, SWE phi-share weight derivation, loss classification) and of
``archive_prelaunch.py`` (content hashing, hash comparison, Category 3
observation-metadata decoding).
"""

import importlib.util
import json
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
def audit():
    return _load("objective_audit")


@pytest.fixture(scope="module")
def arch():
    return _load("archive_prelaunch")


DATES = pd.date_range("2015-01-01", periods=6, freq="D")


def _members():
    return pd.DataFrame(
        {
            "ssebop": [0.80, np.nan, 0.60, 0.90, 0.50, np.nan],
            "ptjpl": [0.70, 0.65, np.nan, 0.80, 0.40, np.nan],
        },
        index=DATES,
    )


def test_parse_obs_name(audit):
    assert audit.parse_obs_name("oname:obs_etf_us-ne1_otype:arr_i:123_j:0") == (
        "etf",
        "us-ne1",
        123,
    )
    assert audit.parse_obs_name("oname:obs_swe_ar-cca_otype:arr_i:0_j:0") == ("swe", "ar-cca", 0)
    with pytest.raises(ValueError):
        audit.parse_obs_name("oname:pred_etf_x_otype:arr_i:1_j:0")


def test_reconstruct_weights_member_count_and_floor(audit):
    eto = pd.Series([3.0, 3.0, 3.0, 0.5, 1.0, 3.0], index=DATES)
    out = audit.reconstruct_etf_weights(
        _members(), eto, spread_floor=0.05, min_members=2, eto_floor=1.0, fixed_sd=0.33
    )
    # the all-NaN date is not a capture
    assert list(out.index) == list(DATES[:5])
    assert out.member_count.tolist() == [2, 1, 1, 2, 2]
    # two-member date: target = mean, weight = target / (sample sd + floor)
    sd = np.std([0.80, 0.70], ddof=1)
    assert out.loc[DATES[0], "target"] == pytest.approx(0.75)
    assert out.loc[DATES[0], "weight"] == pytest.approx(0.75 / (sd + 0.05))
    assert out.loc[DATES[0], "standard_deviation"] == pytest.approx(sd + 0.05)
    # one-member dates: target is the member, weight zero, noise sd falls back to fixed_sd
    assert out.loc[DATES[1], "target"] == pytest.approx(0.65)
    assert out.loc[DATES[1], "weight"] == 0.0
    assert out.loc[DATES[1], "standard_deviation"] == pytest.approx(0.33)
    assert not out.loc[DATES[1], "eligible"]
    # two members but ETo below the floor: zero weight, target kept, flagged
    assert out.loc[DATES[3], "weight"] == 0.0
    assert out.loc[DATES[3], "target"] == pytest.approx(0.85)
    assert bool(out.loc[DATES[3], "eto_floor_excluded"])
    # ETo exactly at the floor passes (inclusive)
    assert out.loc[DATES[4], "weight"] > 0
    assert not out.loc[DATES[4], "eto_floor_excluded"]


def test_reconstruct_weights_floor_disabled_and_missing_eto(audit):
    eto = pd.Series([3.0, 3.0, 3.0, 0.5, 1.0, 3.0], index=DATES)
    out = audit.reconstruct_etf_weights(
        _members(), eto, spread_floor=0.05, min_members=2, eto_floor=None, fixed_sd=0.33
    )
    assert out.loc[DATES[3], "weight"] > 0
    assert not out.eto_floor_excluded.any()
    eto_missing = eto.copy()
    eto_missing.iloc[0] = np.nan
    with pytest.raises(ValueError, match="daily ETo missing"):
        audit.reconstruct_etf_weights(
            _members(), eto_missing, spread_floor=0.05, min_members=2, eto_floor=1.0, fixed_sd=0.33
        )


def test_swe_expected_weights_phi_share(audit):
    swe = np.array([5.0, 50.0, 200.0])
    etf_w = np.array([2.0, 3.0, 0.0, 4.0])
    etf_sd = np.array([0.1, 0.2, 0.33, 0.15])
    w, sd, c = audit.swe_expected_weights(
        swe, etf_w, etf_sd, sd_frac=0.3, sd_floor=10.0, phi_share=0.15
    )
    assert sd.tolist() == [10.0, 15.0, 60.0]  # floor binds for the small values
    etf_phi = float(((etf_w * etf_sd) ** 2).sum())
    swe_phi = float(((w * sd) ** 2).sum())
    assert swe_phi == pytest.approx(0.15 / 0.85 * etf_phi)
    assert c == pytest.approx(w[0] * sd[0])
    w0, _, c0 = audit.swe_expected_weights(
        swe, np.zeros(2), np.ones(2), sd_frac=0.3, sd_floor=10.0, phi_share=0.15
    )
    assert c0 == 0.0 and not w0.any()


def test_classify_loss(audit):
    base = {
        "member_count": 1,
        "eto": 3.0,
        "eto_floor_excluded": False,
        "corrected": np.nan,
        "status": "",
    }
    assert (
        audit.classify_loss(
            pd.Series({**base, "eto_floor_excluded": True, "member_count": 2}), 0.05, 2.0
        )
        == "eto_floor"
    )
    assert audit.classify_loss(pd.Series({**base, "corrected": 2.4}), 0.05, 2.0) == "ingest_ceiling"
    assert audit.classify_loss(pd.Series({**base, "corrected": 0.01}), 0.05, 2.0) == "ingest_floor"
    assert audit.classify_loss(pd.Series(base), 0.05, 2.0) == "unexplained_no_ledger_row"
    assert (
        audit.classify_loss(pd.Series({**base, "corrected": 0.7}), 0.05, 2.0)
        == "unexplained_member_missing"
    )
    assert audit.classify_loss(pd.Series({**base, "member_count": 2}), 0.05, 2.0).startswith(
        "unexplained"
    )
    assert audit.EXPLAINED_LOSS == {"eto_floor", "ingest_ceiling", "ingest_floor"}


def test_hash_helpers(arch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("abc")
    assert arch.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    x = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    assert arch.array_content_sha256(x) == arch.array_content_sha256(x.copy())
    assert arch.array_content_sha256(x) != arch.array_content_sha256(x.astype(np.float64))
    s = np.array(["US-Ne1", "US-Ne2"], dtype=object)
    assert arch.array_content_sha256(s) == arch.array_content_sha256(np.array(["US-Ne1", "US-Ne2"]))
    assert arch.compare_hashes({"a": "1", "b": "2"}, {"a": "1", "b": "3", "c": "4"}) == [
        "changed: b",
        "new (not recorded): c",
    ]
    assert arch.compare_hashes({"a": "1"}, {}) == ["missing now: a"]
    assert arch.compare_hashes({"a": "1"}, {"a": "1"}) == []


def test_observation_metadata_decoding(arch):
    rows = pd.DataFrame(
        {
            "fid": ["US-Ne1", "US-Ne1"],
            "batch": ["batch_000", "batch_000"],
            "date": ["2013-01-01", "2013-01-03"],
            "target": [0.75, 0.65],
            "member_count": [2, 1],
            "member_std": [0.0707, np.nan],
            "eto": [3.0, 2.0],
            "eto_floor_excluded": [False, False],
            "weight": [6.2, 0.0],
            "weight_pst": [6.2, 0.0],
            "sd_pst": [0.1207, 0.33],
            "ssebop": [0.80, np.nan],
            "ptjpl": [0.70, 0.65],
        }
    )
    resolved = {
        "start_date": "2013-01-01",
        "etf_target_instrument": "landsat",
        "etf_target_model": "ensemble",
        "mask": "no_mask",
    }
    meta = arch.observation_metadata(rows, ["ssebop", "ptjpl"], resolved)
    assert meta.obsnme.tolist() == [
        "oname:obs_etf_us-ne1_otype:arr_i:0_j:0",
        "oname:obs_etf_us-ne1_otype:arr_i:2_j:0",
    ]
    assert json.loads(meta.raw_member_values.iloc[0]) == [0.8, 0.7]
    assert json.loads(meta.raw_member_values.iloc[1]) == [None, 0.65]
    assert meta.final_weight.tolist() == [6.2, 0.0]
    assert (meta.weight_formula == arch.WEIGHT_FORMULA).all()
    for col in (
        "obsnme",
        "site",
        "date",
        "sensor",
        "model",
        "mask_mode",
        "target_etf",
        "raw_member_values",
        "member_count",
        "ensemble_std",
        "mad_included",
        "eto_correction_factor",
        "raw_weight",
        "final_weight",
        "weight_formula",
    ):
        assert col in meta.columns
