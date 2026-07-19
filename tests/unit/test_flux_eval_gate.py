"""E1 (Example 4) evaluation metric-gate tests.

Regression guard for the monthly-gate bug: `evaluate_monthly()` used to admit
sites at 6 paired months while `calc_metrics()` returned NaN below 10, leaving
all-NaN metric rows in the monthly CSV (and an inflated cohort denominator).
The fix routes every admission gate (daily, monthly, ETf) and the metric
routine through one constant, `MIN_OBS_FOR_METRICS`. These tests pin the
boundary so the two thresholds cannot silently drift apart again.

`examples/4_Flux_Network/evaluate.py` is a script, not a package module, so it
is loaded by path.
"""

import importlib.util
from pathlib import Path

import numpy as np

_EVAL_PATH = Path(__file__).resolve().parents[2] / "examples" / "4_Flux_Network" / "evaluate.py"


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("e1_evaluate", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_evaluate()


def _paired(n, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(3.0, 1.0, size=n)
    mod = obs + rng.normal(0.0, 0.2, size=n)
    return obs, mod


def test_metric_floor_is_ten():
    # The single source of truth every gate references.
    assert ev.MIN_OBS_FOR_METRICS == 10


def test_below_floor_returns_nan():
    obs, mod = _paired(ev.MIN_OBS_FOR_METRICS - 1)
    m = ev.calc_metrics(obs, mod)
    assert m["n"] == ev.MIN_OBS_FOR_METRICS - 1
    for k in ("r2", "r", "rmse", "bias", "kge"):
        assert np.isnan(m[k]), f"{k} should be NaN below the floor"


def test_at_floor_returns_finite():
    obs, mod = _paired(ev.MIN_OBS_FOR_METRICS)
    m = ev.calc_metrics(obs, mod)
    assert m["n"] == ev.MIN_OBS_FOR_METRICS
    for k in ("r2", "r", "rmse", "bias", "kge"):
        assert np.isfinite(m[k]), f"{k} should be finite at the floor"


def test_nan_pairs_counted_after_masking():
    # The monthly-gate failure mode: enough raw rows, but non-finite entries
    # drop the *finite* count below the floor -> must yield NaN, never a
    # spuriously "finite" row. calc_metrics masks before counting.
    obs, mod = _paired(ev.MIN_OBS_FOR_METRICS + 2)
    obs[:3] = np.nan  # 9 finite pairs remain, below the floor of 10
    m = ev.calc_metrics(obs, mod)
    assert m["n"] == ev.MIN_OBS_FOR_METRICS - 1
    assert np.isnan(m["r2"])


# --- Source-level regression guards ---------------------------------------
# The gate bug and the manuscript-facing metric cleanup are structural: they
# can regress without changing calc_metrics. Guard the evaluate.py source so a
# silent-drop or a reintroduced win-rate/threshold cannot slip back in.

_SRC = _EVAL_PATH.read_text()


def test_no_stale_numeric_monthly_gate():
    # The old bug: a bare `< 6` monthly gate diverging from the metric floor.
    # Every admission gate must route through the constant, not a literal.
    assert "< 6" not in _SRC and ">= 6" not in _SRC
    # daily + monthly + etf gates and the metric routine all reference it.
    assert _SRC.count("MIN_OBS_FOR_METRICS") >= 6


def test_monthly_floor_records_exclusion():
    # The monthly path must LOG the sites it drops at the metric floor and the
    # <30-paired-days gate, so the ledger reconciles configured - excluded =
    # scored. A silent `continue` is exactly the reconciliation bug.
    assert "below_monthly_metric_floor" in _SRC
    assert "below_30_paired_days" in _SRC
    # The monthly ledger goes to its own file (the daily path uses the default),
    # so the two cohorts do not clobber one another.
    assert "evaluation_sites_excluded_monthly.csv" in _SRC


def test_canonical_exclusion_is_recorded():
    # MB_Pch is dropped before the loop by apply_exclusions; both eval paths
    # must still record it so the ledger is complete.
    assert "canonical_exclusion_data_quality" in _SRC


def test_win_rate_prints_removed():
    # Manuscript-facing cleanup: win-rate reporting is gone from the evaluator.
    assert "win rate" not in _SRC.lower()
