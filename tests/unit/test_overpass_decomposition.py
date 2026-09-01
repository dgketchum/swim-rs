"""Unit tests for the Run 22 overpass/non-overpass decomposition (Example 5).

Synthetic-data tests for the split invariants on the ETf-first footing:

1. raw benchmark dates, not calibration flags, define the direct subset;
2. the benchmark is reconstructed ETf-first on the common ETo basis — it
   visibly differs from direct-ET interpolation under varying ETo;
3. temporal support follows the Volk ±32-day rule with openet-core
   semantics (one-sided flat fill inside the window, NaN outside);
4. the subsets are disjoint and their union equals the paired all-days record;
5. both models are scored on identical dates;
6. a site below the 10-date subset threshold is explicitly excluded.

The script lives outside the swimrs package and is imported by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "5_Flux_Ensemble"
    / "overpass_decomposition.py"
)


@pytest.fixture(scope="module")
def od():
    spec = importlib.util.spec_from_file_location("overpass_decomposition", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def synthetic_site():
    """Frozen-style site frame, a raw benchmark ET series, and a varying ETo.

    Calibration captures (is_overpass) are deliberately placed on DIFFERENT
    dates than the raw benchmark values, so any leakage of the calibration
    flag into the split is detectable. The ETo varies strongly (7-day
    sawtooth), so an invalid direct-ET interpolation is numerically
    distinguishable from the ETf-first reconstruction.
    """
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    rng = np.random.default_rng(7)
    frozen = pd.DataFrame(
        {
            "swim_ET": rng.uniform(0.5, 5.0, len(idx)),
            "flux_ET": rng.uniform(0.5, 5.0, len(idx)),
            "is_overpass": False,
        },
        index=idx,
    )
    # calibration captures every 7 days from Jan 3 — NOT the benchmark dates
    calib_dates = pd.date_range("2020-01-03", "2020-03-27", freq="7D")
    frozen.loc[calib_dates, "is_overpass"] = True

    # raw benchmark ET: finite every 8 days from Jan 10 through Mar 22 only
    raw_dates = pd.date_range("2020-01-10", "2020-03-22", freq="8D")
    raw = pd.Series(np.nan, index=idx)
    raw.loc[raw_dates] = np.linspace(1.0, 3.0, len(raw_dates))

    # strongly varying, strictly positive ETo (sawtooth, period 7 days)
    eto = pd.Series(2.0 + 1.5 * (np.arange(len(idx)) % 7) / 6.0, index=idx)
    return frozen, raw, raw_dates, calib_dates, eto


def _paired(od, frozen, raw, eto):
    daily, direct, recon = od.reconstruct_benchmark(raw, eto, label="SYN")
    paired = od.classify_paired(frozen, daily, direct)
    od.check_date_semantics(paired, recon, "SYN")
    return paired, daily, direct, recon


def test_raw_benchmark_dates_define_direct_subset(od, synthetic_site):
    frozen, raw, raw_dates, calib_dates, eto = synthetic_site
    paired, _, direct, _ = _paired(od, frozen, raw, eto)
    over = paired[paired["is_benchmark_overpass"]]
    assert set(over.index) == set(raw_dates)
    assert set(direct) == set(raw_dates)
    # calibration captures must NOT leak into the split definition
    assert set(over.index) != set(calib_dates)
    assert not set(calib_dates).issubset(set(over.index))


def test_etf_first_differs_from_direct_et_interpolation(od, synthetic_site):
    frozen, raw, raw_dates, _, eto = synthetic_site
    _, daily, _, _ = _paired(od, frozen, raw, eto)
    finite = raw[np.isfinite(raw.values)]
    # the superseded (invalid) construction: linear interpolation of ET itself
    span = pd.date_range(raw_dates.min(), raw_dates.max(), freq="D")
    direct_et = finite.reindex(span).interpolate(method="linear")
    between = span.difference(raw_dates)
    max_gap = (daily.loc[between] - direct_et.loc[between]).abs().max()
    assert max_gap > 0.05, "ETf-first must diverge from direct-ET under varying ETo"
    # both reproduce the raw ET exactly on capture dates
    assert np.allclose(daily.loc[raw_dates].values, finite.values)


def test_volk_window_limits_support(od, synthetic_site):
    frozen, _, _, _, eto = synthetic_site
    idx = frozen.index
    raw = pd.Series(np.nan, index=idx)
    raw.loc[pd.Timestamp("2020-01-10")] = 2.0
    raw.loc[pd.Timestamp("2020-03-30")] = 3.0  # 80-day gap
    paired, daily, _, recon = _paired(od, frozen, raw, eto)
    # interior days more than 32 d from both captures are unsupported: unpaired
    dead = pd.date_range("2020-02-12", "2020-02-26", freq="D")
    assert not np.isfinite(daily.loc[dead].values).any()
    assert len(paired.index.intersection(dead)) == 0
    assert (recon.support_class.loc[dead] == "unsupported").all()
    # one-sided flat fill holds ETf (not ET) flat within the window
    etf0 = 2.0 / eto.loc["2020-01-10"]
    assert daily.loc["2020-01-15"] == pytest.approx(etf0 * eto.loc["2020-01-15"])
    # flat fill also extends before the first capture, inside the window
    assert daily.loc["2020-01-01"] == pytest.approx(etf0 * eto.loc["2020-01-01"])


def test_flat_fill_days_are_non_overpass(od, synthetic_site):
    frozen, raw, raw_dates, _, eto = synthetic_site
    paired, daily, _, recon = _paired(od, frozen, raw, eto)
    tail = pd.date_range("2020-03-23", "2020-03-31", freq="D")
    assert set(tail).issubset(set(paired.index))
    assert (recon.support_class.loc[tail] == "flat_fill").all()
    assert not paired.loc[tail, "is_benchmark_overpass"].any()
    # ETf is held flat past the last capture; ET still varies with daily ETo
    etf_last = raw.loc[raw_dates[-1]] / eto.loc[raw_dates[-1]]
    assert np.allclose(daily.loc[tail].values, (etf_last * eto.loc[tail]).values)


def test_subsets_disjoint_and_union_equals_all_days(od, synthetic_site):
    frozen, raw, _, _, eto = synthetic_site
    paired, _, _, _ = _paired(od, frozen, raw, eto)
    subs = od.subset_frames(paired)
    over, non, all_days = subs["overpass"], subs["non_overpass"], subs["all_days"]
    assert len(over.index.intersection(non.index)) == 0
    assert set(over.index) | set(non.index) == set(all_days.index)
    assert len(over) + len(non) == len(all_days)


def test_both_models_scored_on_identical_dates(od, synthetic_site):
    frozen, raw, _, _, eto = synthetic_site
    # knock a flux value out inside the benchmark span: that date must drop
    # from the paired record for BOTH models, not just one
    frozen = frozen.copy()
    frozen.loc["2020-02-01", "flux_ET"] = np.nan
    paired, _, _, _ = _paired(od, frozen, raw, eto)
    assert pd.Timestamp("2020-02-01") not in paired.index
    for sdf in od.subset_frames(paired).values():
        obs = sdf["flux"].values
        m_swim = od.calc_metrics(obs, sdf["swim"].values)
        m_openet = od.calc_metrics(obs, sdf["openet"].values)
        assert m_swim["n"] == m_openet["n"] == len(sdf)


def test_below_threshold_subset_is_excluded(od, synthetic_site):
    frozen, _, _, _, eto = synthetic_site
    # only 6 raw benchmark dates -> overpass subset below the 10-date minimum
    idx = frozen.index
    raw = pd.Series(np.nan, index=idx)
    sparse_dates = pd.date_range("2020-01-10", periods=6, freq="10D")
    raw.loc[sparse_dates] = 2.0
    paired, _, _, _ = _paired(od, frozen, raw, eto)
    over = od.subset_frames(paired)["overpass"]
    assert len(over) == 6 < od.MIN_PAIRED
    m = od.calc_metrics(over["flux"].values, over["swim"].values)
    assert m["n"] == 6
    for k in od.METRIC_KEYS:
        assert np.isnan(m[k]), f"{k} must be NaN below the {od.MIN_PAIRED}-date minimum"


def test_january_source_rejected(od, tmp_path):
    jan = tmp_path / "data" / "openet_flux" / "daily_data"
    jan.mkdir(parents=True)
    with pytest.raises(ValueError, match="openet_flux_2pt1"):
        od.assert_may_source(jan)
    may = tmp_path / "data" / "openet_flux_2pt1" / "daily_data"
    may.mkdir(parents=True)
    assert od.assert_may_source(may) == may


def test_kge_is_gupta_2009_form(od):
    rng = np.random.default_rng(3)
    obs = rng.uniform(1, 5, 50)
    mod = obs * 1.1 + rng.normal(0, 0.2, 50)
    m = od.calc_metrics(obs, mod)
    r = np.corrcoef(obs, mod)[0, 1]
    alpha = mod.std() / obs.std()
    beta = mod.mean() / obs.mean()
    expected = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    assert m["kge"] == pytest.approx(expected, abs=1e-12)
    # NSE key is 'nse' (1 - SSE/SST); no 'r2' key in publication-facing output
    assert "nse" in m and "r2" not in m
