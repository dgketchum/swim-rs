"""Gate logic for the pooled two-arm transpiration-form comparison.

``pooled_arm_compare.decide`` is what determines whether a multi-hour follow-up
calibration gets launched, so its metric directions are worth pinning down:
RMSE lower-is-better, KGE higher-is-better, and MBE compared on **absolute**
value (a signed bias of -0.01 beats +0.30, and naive "lower wins" would get
that backwards).
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

EX5 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "5_Flux_Ensemble",
)


def _load():
    if EX5 not in sys.path:
        sys.path.insert(0, EX5)
    spec = importlib.util.spec_from_file_location(
        "pooled_arm_compare", os.path.join(EX5, "pooled_arm_compare.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _pooled(obs, a, b):
    return {"obs": np.asarray(obs), "a": np.asarray(a), "b": np.asarray(b)}


def test_identical_arms_give_no_wins(mod):
    """Two identical arms must not hand arm A a win on any metric."""
    rng = np.random.default_rng(0)
    obs = rng.normal(3.0, 1.0, 400)
    same = obs + rng.normal(0.0, 0.4, 400)
    p = _pooled(obs, same, same)
    table, wins_a, passed = mod.decide("A", "B", p, p)
    assert wins_a == 0
    assert not passed
    assert set(table["winner"]) == {"B"}


def test_clearly_better_arm_a_passes(mod):
    """Arm A closer to obs on every count wins all six and passes the gate."""
    rng = np.random.default_rng(1)
    obs = rng.normal(3.0, 1.0, 400)
    good = obs + rng.normal(0.0, 0.2, 400)
    bad = obs + rng.normal(0.8, 1.2, 400)
    table, wins_a, passed = mod.decide("A", "B", _pooled(obs, good, bad), _pooled(obs, good, bad))
    assert wins_a == 6
    assert passed
    assert set(table["winner"]) == {"A"}


def test_mbe_compared_on_absolute_value(mod):
    """A small negative bias must beat a large positive one."""
    obs = np.full(200, 4.0)
    a = obs - 0.05  # MBE -0.05
    b = obs + 0.90  # MBE +0.90
    table, _, _ = mod.decide("A", "B", _pooled(obs, a, b), _pooled(obs, a, b))
    mbe = table[table["metric"] == "MBE"]
    assert (mbe["winner"] == "A").all(), "abs(MBE) must decide, not the signed value"


def test_gate_threshold_is_four_of_six(mod):
    """Exactly 4 wins passes; 3 does not."""
    rng = np.random.default_rng(2)
    obs = rng.normal(3.0, 1.0, 400)
    # Arm A better daily, arm B better monthly -> 3/6, must fail.
    a_good = obs + rng.normal(0.0, 0.2, 400)
    b_bad = obs + rng.normal(0.6, 1.1, 400)
    _, wins_split, passed_split = mod.decide(
        "A", "B", _pooled(obs, a_good, b_bad), _pooled(obs, b_bad, a_good)
    )
    assert wins_split == 3
    assert not passed_split

    # Sanity: the documented rule is >= 4.
    assert mod.decide("A", "B", _pooled(obs, a_good, b_bad), _pooled(obs, a_good, b_bad))[2]


def test_metric_directions_declared(mod):
    """The direction table is the single source of truth for 'who won'."""
    directions = dict(mod.METRICS)
    assert directions == {"rmse": "lower", "bias": "abs_lower", "kge": "higher"}
