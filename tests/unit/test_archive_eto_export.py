"""Archive export must record the ETo that SWIM actually consumed.

Guards the archive_run.py fix for the provenance error found in the run22
archive: ``site_daily_timeseries`` exported raw gridMET ``eto`` while the
model-input loader prefers ``eto_corr`` (swimrs.process.input). The exported
``eto`` column must follow the same preference — corrected when present —
with raw gridMET retained under ``eto_raw`` for provenance only.

The script lives outside the swimrs package and is imported by path (it
imports its sibling evaluate.py, so that directory goes on sys.path).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"


@pytest.fixture(scope="module")
def ar():
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("archive_run", EXAMPLE_DIR / "archive_run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


@pytest.fixture()
def arrays():
    rng = np.random.default_rng(11)
    raw = rng.uniform(1.0, 8.0, size=(30, 3))
    corr = raw * 0.9  # corrected series distinguishably different from raw
    return raw, corr


def test_export_prefers_eto_corr(ar, arrays):
    raw, corr = arrays
    root = {"meteorology/gridmet/eto": raw, "meteorology/gridmet/eto_corr": corr}
    active, retained_raw = ar.active_refet_arrays(root)
    assert np.array_equal(active, corr), "active eto must be eto_corr when present"
    assert np.array_equal(retained_raw, raw), "raw gridMET retained as eto_raw"


def test_export_falls_back_to_raw_eto(ar, arrays):
    raw, _ = arrays
    root = {"meteorology/gridmet/eto": raw}
    active, retained_raw = ar.active_refet_arrays(root)
    assert np.array_equal(active, raw)
    assert retained_raw is None, "no eto_raw column when raw is already active"


def test_export_matches_model_input_preference(ar, arrays):
    """Same selection rule as swimrs.process.input: {refet_type}_corr first."""
    raw, corr = arrays
    root = {"meteorology/gridmet/etr": raw, "meteorology/gridmet/etr_corr": corr}
    active, retained_raw = ar.active_refet_arrays(root, refet_type="etr")
    assert np.array_equal(active, corr)
    assert np.array_equal(retained_raw, raw)
