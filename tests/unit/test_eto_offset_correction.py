"""Tests for the next-day ETo divisor correction (Example 5 extract tables)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CB = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble" / "container_build"


@pytest.fixture(scope="module")
def mod():
    if str(CB) not in sys.path:
        sys.path.insert(0, str(CB))
    spec = importlib.util.spec_from_file_location(
        "correct_eto_offset", CB / "correct_eto_offset.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _eto(days=12, sites=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-06-01", periods=days, freq="D")
    return pd.DataFrame(rng.uniform(3, 8, size=(days, len(sites))), index=idx, columns=list(sites))


def _etf(eto, rows):
    df = pd.DataFrame(np.nan, index=eto.index[: len(eto) - 1], columns=eto.columns)
    for (day, site), v in rows.items():
        df.loc[eto.index[day], site] = v
    return df


def test_factor_identity_and_nan_pattern(mod):
    eto = _eto()
    df = _etf(eto, {(0, "A"): 0.8, (3, "A"): 1.1, (3, "B"): 0.6, (9, "B"): 0.4})
    corrected, factor = mod.correct_frame(df, eto)
    assert corrected.isna().equals(df.isna())
    for (day, site), v in {(0, "A"): 0.8, (3, "A"): 1.1, (3, "B"): 0.6, (9, "B"): 0.4}.items():
        d = eto.index[day]
        expected = v * eto.loc[d + pd.Timedelta(days=1), site] / eto.loc[d, site]
        assert corrected.loc[d, site] == pytest.approx(expected, rel=1e-12)
        assert factor.loc[d, site] == pytest.approx(expected / v, rel=1e-12)


def test_missing_next_day_eto_raises(mod):
    eto = _eto()
    df = _etf(eto, {(5, "A"): 0.9})
    eto.loc[eto.index[6], "A"] = np.nan
    with pytest.raises(ValueError, match="lack a positive ETo"):
        mod.correct_frame(df, eto)


def test_non_positive_same_day_eto_raises(mod):
    eto = _eto()
    df = _etf(eto, {(5, "B"): 0.9})
    eto.loc[eto.index[5], "B"] = 0.0
    with pytest.raises(ValueError, match="lack a positive ETo"):
        mod.correct_frame(df, eto)


def test_capture_past_eto_range_raises(mod):
    eto = _eto()
    df = pd.DataFrame(np.nan, index=eto.index, columns=eto.columns)
    df.loc[eto.index[-1], "A"] = 0.7  # no d+1 in the table
    with pytest.raises(ValueError, match="lack a positive ETo"):
        mod.correct_frame(df, eto)


def test_no_filter_applied_and_crossings_counted(mod):
    idx = pd.date_range("2016-06-01", periods=4, freq="D")
    eto = pd.DataFrame({"A": [4.0, 8.0, 2.0, 3.0], "B": [5.0, 5.0, 5.0, 5.0]}, index=idx)
    df = pd.DataFrame(np.nan, index=idx[:3], columns=["A", "B"])
    df.loc[idx[0], "A"] = 0.04  # factor 2 -> 0.08, regained from below
    df.loc[idx[1], "A"] = 1.5  # factor 0.25 -> 0.375, stays valid
    df.loc[idx[2], "A"] = 1.5  # factor 1.5 -> 2.25, lost above
    df.loc[idx[0], "B"] = 0.06  # factor 1 -> unchanged
    corrected, _ = mod.correct_frame(df, eto)
    assert corrected.loc[idx[2], "A"] == pytest.approx(2.25)  # not filtered here
    bc = mod.bound_crossings(df, corrected)
    assert bc["regained_from_below"] == 1
    assert bc["regained_from_above"] == 0
    assert bc["lost_above"] == 1
    assert bc["lost_below"] == 0
    assert bc["n_valid_window"] == 4


def test_directory_round_trip(mod, tmp_path):
    eto = _eto(days=8)
    etf_dir = tmp_path / "etf"
    etf_dir.mkdir()
    eto_wide = eto.T
    eto_wide.columns = [d.strftime("%Y%m%d") for d in eto_wide.columns]
    eto_wide.index.name = "site_id"
    eto_csv = tmp_path / "openet_eto.csv"
    eto_wide.to_csv(eto_csv)

    stored = {}
    for model in mod.MODELS:
        df = _etf(eto, {(1, "A"): 0.5, (4, "B"): 1.2})
        mod._write_wide(df, etf_dir / f"{model}_etf_no_mask.csv")
        stored[model] = df
        with open(etf_dir / f"{model}_summary.json", "w") as f:
            json.dump({"model": model, "et_denominated": model in mod.AFFECTED_MODELS}, f)

    out = tmp_path / "fixed"
    report = mod.correct_directory(etf_dir, eto_csv, out)
    assert set(report) == set(mod.AFFECTED_MODELS)

    for model in mod.MODELS:
        got = mod._read_etf_csv_max(out / f"{model}_etf_no_mask.csv")
        if model in mod.AFFECTED_MODELS:
            expected, _ = mod.correct_frame(stored[model], eto)
            summary = json.load(open(out / f"{model}_summary.json"))
            assert "eto_offset_correction" in summary
            side = json.load(open(out / f"{model}_eto_offset_correction.json"))
            assert side["factor"]["n"] == 2
            assert len(side["corrected_csv_sha256"]) == 64
        else:
            expected = stored[model]
            assert (out / f"{model}_etf_no_mask.csv").read_bytes() == (
                etf_dir / f"{model}_etf_no_mask.csv"
            ).read_bytes()
        pd.testing.assert_frame_equal(
            got.dropna(how="all"), expected.dropna(how="all"), check_names=False, check_freq=False
        )

    with pytest.raises(FileExistsError):
        mod.correct_directory(etf_dir, eto_csv, out)
