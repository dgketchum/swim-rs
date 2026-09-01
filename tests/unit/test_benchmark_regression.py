"""Frozen-value regression test for the E1 benchmark evidence rebuild.

Replays the ETf-first May v2.1 reconstruction and the evaluate.py-semantics
daily/monthly metric rows for three representative cohort sites against
values pinned in ``tests/fixtures/e1_benchmark/expected.json`` (emitted by
``rebuild_e1_benchmark_evidence.py --emit-test-fixture``). Any change to the
reconstruction (window semantics, ETo basis, pairing masks) or the metric
conventions moves these values and fails here. Cohort-level medians/counts
are gated separately by the rebuild script's ``--verify`` (G-VALUES).

The script lives outside the swimrs package and is imported by path.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "examples" / "5_Flux_Ensemble" / "rebuild_e1_benchmark_evidence.py"
FIXTURE = REPO / "tests" / "fixtures" / "e1_benchmark"

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def reb():
    spec = importlib.util.spec_from_file_location("rebuild_e1_benchmark_evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def expected():
    with open(FIXTURE / "expected.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def eto():
    with gzip.open(FIXTURE / "openet_eto_subset.csv.gz", "rt") as f:
        return pd.read_csv(f, index_col=0, parse_dates=True)


def _read_gz(name, index_col):
    with gzip.open(FIXTURE / name, "rt") as f:
        return pd.read_csv(f, index_col=index_col, parse_dates=True)


def _assert_row_matches(row, exp, fid, scale):
    assert set(row) == set(exp), f"{fid} {scale}: key set changed"
    for k, want in exp.items():
        got = row[k]
        if want is None:
            assert isinstance(got, float) and np.isnan(got), f"{fid} {scale} {k}: expected NaN"
        elif isinstance(want, float):
            assert got == pytest.approx(want, abs=1e-9), f"{fid} {scale} {k}"
        else:
            assert got == want, f"{fid} {scale} {k}"


def test_expected_fixture_shape(expected):
    assert len(expected["fids"]) == 3
    assert expected["window_days"] == 32
    assert expected["eto_source"] == "openet_refet/openet_eto.csv"


@pytest.mark.parametrize("fid_idx", [0, 1, 2])
def test_daily_site_row_reproduces_pinned_values(reb, expected, eto, fid_idx):
    fid = expected["fids"][fid_idx]
    ts = _read_gz(f"{fid}_timeseries.csv.gz", "date")
    may_daily = _read_gz(f"{fid}_openet_daily.csv.gz", "DATE")
    recons = reb.reconstruct_site_series(may_daily, eto[fid].astype("float64"), fid)
    row = reb.daily_site_row(fid, ts["swim_ET"].astype(float), ts["flux_ET"].astype(float), recons)
    _assert_row_matches(row, expected[fid]["daily"], fid, "daily")


@pytest.mark.parametrize("fid_idx", [0, 1, 2])
def test_monthly_site_row_reproduces_pinned_values(reb, expected, fid_idx):
    fid = expected["fids"][fid_idx]
    exp = expected[fid]["monthly"]
    ts = _read_gz(f"{fid}_timeseries.csv.gz", "date")
    may_monthly = _read_gz(f"{fid}_openet_monthly.csv.gz", "DATE")
    row = reb.monthly_site_row(
        fid, ts["swim_ET"].astype(float), ts["flux_ET"].astype(float), may_monthly
    )
    if exp is None:
        assert row is None, f"{fid}: expected monthly exclusion"
    else:
        _assert_row_matches(row, exp, fid, "monthly")


def test_capture_identity_on_fixture_sites(reb, expected, eto):
    """Reconstructed ET reproduces the May capture ET exactly (G-IDENT)."""
    for fid in expected["fids"]:
        may_daily = _read_gz(f"{fid}_openet_daily.csv.gz", "DATE")
        recons = reb.reconstruct_site_series(may_daily, eto[fid].astype("float64"), fid)
        assert "ensemble" in recons
        for name, recon in recons.items():
            assert recon.identity_max_abs_err <= 1e-10, f"{fid}:{name}"
