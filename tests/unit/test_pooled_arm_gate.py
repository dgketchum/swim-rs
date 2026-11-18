"""Metric directions of the pooled two-arm vegetation-formulation comparison.

``pooled_arm_compare.pooled_table`` labels the better arm on each pooled metric:
RMSE lower-is-better, KGE higher-is-better, and MBE compared on **absolute**
value (a signed bias of -0.01 beats +0.30, and naive "lower wins" would get
that backwards). The table reports the metrics; it applies no pass/fail rule.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

EX6 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "6_Flux_International",
)


def _load():
    # The script imports its sibling evaluate.py; other tests load the Example 5
    # evaluate.py under the same module name, so isolate the import.
    saved = sys.modules.pop("evaluate", None)
    sys.path.insert(0, EX6)
    try:
        spec = importlib.util.spec_from_file_location(
            "pooled_arm_compare", os.path.join(EX6, "pooled_arm_compare.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(EX6)
        sys.modules.pop("evaluate", None)
        if saved is not None:
            sys.modules["evaluate"] = saved
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _pooled(obs, a, b):
    return {"obs": np.asarray(obs), "a": np.asarray(a), "b": np.asarray(b)}


def test_table_has_six_rows_and_no_verdict(mod):
    """Three metrics at two scales, returned as a table and nothing else."""
    rng = np.random.default_rng(3)
    obs = rng.normal(3.0, 1.0, 400)
    p = _pooled(obs, obs + rng.normal(0.0, 0.2, 400), obs + rng.normal(0.5, 0.9, 400))
    table = mod.pooled_table("A", "B", p, p)
    assert list(table["scale"]) == ["daily"] * 3 + ["monthly"] * 3
    assert list(table["metric"]) == ["RMSE", "MBE", "KGE"] * 2
    assert not {"passed", "wins_a", "gate"} & set(table.columns)


def test_identical_arms_give_no_wins(mod):
    """Two identical arms must not label arm A the better one on any metric."""
    rng = np.random.default_rng(0)
    obs = rng.normal(3.0, 1.0, 400)
    same = obs + rng.normal(0.0, 0.4, 400)
    p = _pooled(obs, same, same)
    table = mod.pooled_table("A", "B", p, p)
    assert set(table["winner"]) == {"B"}
    assert (table["delta_a_minus_b"] == 0).all()


def test_clearly_better_arm_a_is_labelled_on_every_metric(mod):
    """Arm A closer to obs on every count is the better arm in all six rows."""
    rng = np.random.default_rng(1)
    obs = rng.normal(3.0, 1.0, 400)
    good = obs + rng.normal(0.0, 0.2, 400)
    bad = obs + rng.normal(0.8, 1.2, 400)
    table = mod.pooled_table("A", "B", _pooled(obs, good, bad), _pooled(obs, good, bad))
    assert set(table["winner"]) == {"A"}


def test_mbe_compared_on_absolute_value(mod):
    """A small negative bias must beat a large positive one."""
    obs = np.full(200, 4.0)
    a = obs - 0.05  # MBE -0.05
    b = obs + 0.90  # MBE +0.90
    table = mod.pooled_table("A", "B", _pooled(obs, a, b), _pooled(obs, a, b))
    mbe = table[table["metric"] == "MBE"]
    assert (mbe["winner"] == "A").all(), "abs(MBE) must decide, not the signed value"


def test_scales_are_scored_independently(mod):
    """Arm A better daily and arm B better monthly is reported as exactly that."""
    rng = np.random.default_rng(2)
    obs = rng.normal(3.0, 1.0, 400)
    a_good = obs + rng.normal(0.0, 0.2, 400)
    b_bad = obs + rng.normal(0.6, 1.1, 400)
    table = mod.pooled_table("A", "B", _pooled(obs, a_good, b_bad), _pooled(obs, b_bad, a_good))
    assert set(table.loc[table["scale"] == "daily", "winner"]) == {"A"}
    assert set(table.loc[table["scale"] == "monthly", "winner"]) == {"B"}


def test_metric_directions_declared(mod):
    """The direction table is the single source of truth for the better arm."""
    directions = dict(mod.METRICS)
    assert directions == {"rmse": "lower", "bias": "abs_lower", "kge": "higher"}
