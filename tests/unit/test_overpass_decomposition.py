"""Unit tests for the Run 22 overpass/non-overpass decomposition (Example 5).

Synthetic-data tests for the handoff-specified invariants
(examples/5_Flux_Ensemble/notes/run22_overpass_nonoverpass_handoff.md §9):

1. raw benchmark dates, not calibration flags, define the direct subset;
2. interpolation occurs only between the first and last finite raw observations;
3. the subsets are disjoint and their union equals the paired all-days record;
4. both models are scored on identical dates;
5. a site below the 10-date subset threshold is explicitly excluded.

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
    """Frozen-style site frame plus a raw benchmark series.

    Calibration captures (is_overpass) are deliberately placed on DIFFERENT
    dates than the raw benchmark values, so any leakage of the calibration
    flag into the split is detectable. Flux/SWIM extend beyond the benchmark
    support on both ends, so extrapolation would also be detectable.
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

    # raw benchmark: finite every 8 days from Jan 10 through Mar 22 only
    raw_dates = pd.date_range("2020-01-10", "2020-03-22", freq="8D")
    raw = pd.Series(np.nan, index=idx)
    raw.loc[raw_dates] = np.linspace(1.0, 3.0, len(raw_dates))
    return frozen, raw, raw_dates, calib_dates


def _paired(od, frozen, raw):
    daily, direct = od.reconstruct_benchmark(raw)
    paired = od.classify_paired(frozen, daily, direct)
    od.check_date_semantics(paired, direct, "SYN")
    return paired, daily, direct


def test_raw_benchmark_dates_define_direct_subset(od, synthetic_site):
    frozen, raw, raw_dates, calib_dates = synthetic_site
    paired, _, direct = _paired(od, frozen, raw)
    over = paired[paired["is_benchmark_overpass"]]
    assert set(over.index) == set(raw_dates)
    assert set(direct) == set(raw_dates)
    # calibration captures must NOT leak into the split definition
    assert set(over.index) != set(calib_dates)
    assert not set(calib_dates).issubset(set(over.index))


def test_no_extrapolation_outside_raw_support(od, synthetic_site):
    frozen, raw, raw_dates, _ = synthetic_site
    paired, daily, _ = _paired(od, frozen, raw)
    lo, hi = raw_dates.min(), raw_dates.max()
    # flux and SWIM are finite outside [lo, hi], yet no paired date escapes it
    assert paired.index.min() >= lo
    assert paired.index.max() <= hi
    # the interpolated series itself spans exactly first→last finite raw date
    assert daily.index.min() == lo
    assert daily.index.max() == hi
    assert np.isfinite(daily.values).all()


def test_subsets_disjoint_and_union_equals_all_days(od, synthetic_site):
    frozen, raw, _, _ = synthetic_site
    paired, _, _ = _paired(od, frozen, raw)
    subs = od.subset_frames(paired)
    over, non, all_days = subs["overpass"], subs["non_overpass"], subs["all_days"]
    assert len(over.index.intersection(non.index)) == 0
    assert set(over.index) | set(non.index) == set(all_days.index)
    assert len(over) + len(non) == len(all_days)


def test_both_models_scored_on_identical_dates(od, synthetic_site):
    frozen, raw, _, _ = synthetic_site
    # knock a flux value out inside the benchmark span: that date must drop
    # from the paired record for BOTH models, not just one
    frozen = frozen.copy()
    frozen.loc["2020-02-01", "flux_ET"] = np.nan
    paired, _, _ = _paired(od, frozen, raw)
    assert pd.Timestamp("2020-02-01") not in paired.index
    for sdf in od.subset_frames(paired).values():
        obs = sdf["flux"].values
        m_swim = od.calc_metrics(obs, sdf["swim"].values)
        m_openet = od.calc_metrics(obs, sdf["openet"].values)
        assert m_swim["n"] == m_openet["n"] == len(sdf)


def test_below_threshold_subset_is_excluded(od, synthetic_site):
    frozen, _, _, _ = synthetic_site
    # only 6 raw benchmark dates -> overpass subset below the 10-date minimum
    idx = frozen.index
    raw = pd.Series(np.nan, index=idx)
    sparse_dates = pd.date_range("2020-01-10", periods=6, freq="10D")
    raw.loc[sparse_dates] = 2.0
    paired, _, _ = _paired(od, frozen, raw)
    over = od.subset_frames(paired)["overpass"]
    assert len(over) == 6 < od.MIN_PAIRED
    m = od.calc_metrics(over["flux"].values, over["swim"].values)
    assert m["n"] == 6
    for k in od.METRIC_KEYS:
        assert np.isnan(m[k]), f"{k} must be NaN below the {od.MIN_PAIRED}-date minimum"


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
