"""Unit tests for the E2 Phase 8 container health check
(``examples/6_Flux_International/container_build/e2_refooting/phase8_container_health.py``).

Uses small in-memory zarr groups shaped like a SwimContainer to exercise the pure checks:
NaN-aware equality, the grass-CSV replay of the ingestor (bounds, date columns, float32),
the target-identity and regression comparisons, the no-calibration-state guard, and the
baseline -> corrected classifier transition table.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "container_build" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase8_container_health", E2_DIR / "phase8_container_health.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


DATES = pd.date_range("2015-06-01", "2015-06-10", freq="D")


def _container(uids, ssebop, ptjpl, irr, gwsub, calibration=False):
    g = zarr.create_group(store=zarr.storage.MemoryStore())
    g.create_array("time/daily", data=DATES.values.astype("datetime64[ns]"))
    g.create_array("geometry/uid", data=np.array(uids))
    g.create_array("remote_sensing/etf/landsat/ssebop/no_mask", data=np.asarray(ssebop, np.float32))
    g.create_array("remote_sensing/etf/landsat/ptjpl/no_mask", data=np.asarray(ptjpl, np.float32))
    g.create_array("meteorology/era5/eto", data=np.full((len(DATES), len(uids)), 4.0, np.float32))
    g.create_array("derived/dynamics/irr_data", data=np.array([json.dumps(x) for x in irr]))
    g.create_array("derived/dynamics/gwsub_data", data=np.array([json.dumps(x) for x in gwsub]))
    g.attrs["provenance"] = {"events": [{"operation": "create"}, {"operation": "compute"}]}
    if calibration:
        g.create_group("calibration")
        g.attrs["provenance"] = {"events": [{"operation": "simulate"}]}
    return g


def test_nan_equal(mod):
    a = np.array([[1.0, np.nan], [0.5, 2.0]])
    assert mod.nan_equal(a, a.copy()) == (True, 0.0)
    b = a.copy()
    b[1, 1] = 2.5
    same, diff = mod.nan_equal(a, b)
    assert not same and diff == pytest.approx(0.5)
    c = a.copy()
    c[0, 1] = 1.0  # NaN mask differs, finite values agree
    assert mod.nan_equal(a, c) == (False, 0.0)
    assert mod.nan_equal(np.array(["x"]), np.array(["x"])) == (True, 0.0)
    assert mod.nan_equal(np.zeros((2, 2)), np.zeros((3, 2)))[0] is False


def test_grass_replay_applies_ingest_bounds_and_target_identity(mod, tmp_path):
    (tmp_path / "ssebop_etf_grass_A_no_mask_2015.csv").write_text(
        "sid,ETF_20150601,ETF_20150602,ETF_20150603,ETF_20150604,ETF_20140601\n"
        "A,0.5,0.04,2.5,1.3,0.9\n"
    )
    (tmp_path / "ssebop_etf_grass_Z_no_mask_2015.csv").write_text("sid,ETF_20150601\nZ,0.7\n")
    ss = np.full((len(DATES), 1), np.nan)
    ss[0, 0], ss[3, 0] = 0.5, 1.3  # 0.04 (< 0.05) and 2.5 (> 2.0) are dropped by the ingestor
    root = _container(["A"], ss, ss, [{}], [{}])
    expected = mod.expected_ssebop_from_grass(root, str(tmp_path))
    assert expected.dtype == np.float32
    assert np.isfinite(expected).sum() == 2
    check = mod.check_target_identity(root, str(tmp_path))
    assert check["pass"] and check["expected_valid"] == 2 and check["container_valid"] == 2
    ss[3, 0] = 1.31
    root_bad = _container(["A"], ss, ss, [{}], [{}])
    check = mod.check_target_identity(root_bad, str(tmp_path))
    assert not check["pass"] and check["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)


def test_no_calibration_state_guard(mod):
    ss = np.full((len(DATES), 1), 0.5)
    assert mod.check_no_calibration_state(_container(["A"], ss, ss, [{}], [{}]))["pass"]
    bad = mod.check_no_calibration_state(_container(["A"], ss, ss, [{}], [{}], calibration=True))
    assert not bad["pass"]
    assert "calibration/ group present" in bad["problems"]
    assert "simulate events in provenance" in bad["problems"]


def test_regression_and_classifier_transition(mod):
    ss_new = np.full((len(DATES), 2), 0.6)
    pj = np.full((len(DATES), 2), 0.4)
    irr_new = [
        {"2015": {"irr_doys": [160, 170], "irrigated": 1, "f_irr": 0.9}, "fallow_years": []},
        {"2015": {"irr_doys": [], "irrigated": 0, "f_irr": 0.0}, "fallow_years": [2015]},
    ]
    gw = [{"2015": {"subsidized": 0, "f_sub": 0.0, "ratio": 0.5}}] * 2
    new = _container(["A", "B"], ss_new, pj, irr_new, gw)
    # baseline holds the same sites in a different order plus an extra site; PT-JPL identical
    ss_old = np.full((len(DATES), 3), 0.5)
    pj_old = np.full((len(DATES), 3), 0.4)
    irr_old = [
        {"2015": {"irr_doys": [], "irrigated": 0, "f_irr": 0.0}},  # C
        {"2015": {"irr_doys": [], "irrigated": 0, "f_irr": 0.0}},  # B
        {"2015": {"irr_doys": [], "irrigated": 0, "f_irr": 0.0}},  # A (now irrigated)
    ]
    old = _container(["C", "B", "A"], ss_old, pj_old, irr_old, gw + gw[:1])

    reg = mod.check_regression(new, old)
    assert reg["pass"] and reg["n_common_sites"] == 2
    assert reg["arrays"]["remote_sensing/etf/landsat/ptjpl/no_mask"]["identical"]
    assert reg["arrays"]["meteorology/era5/eto"]["identical"]
    assert not reg["ssebop_identical_to_baseline"]
    assert reg["ssebop_max_abs_diff_vs_baseline"] == pytest.approx(0.1, abs=1e-6)

    table, summary = mod.classifier_transition(new, old)
    assert len(table) == 2
    assert summary["irr_transition_counts"] == {"0->1": 1, "0->0": 1}
    assert summary["sites_with_any_year_change"] == ["A"]
    assert summary["ever_irrigated_sites_baseline"] == 0
    assert summary["ever_irrigated_sites_corrected"] == 1
    assert table.set_index("site").loc["A", "n_irr_doys_corrected"] == 2
    assert summary["fallow_site_years_corrected"] == 1
    assert summary["sites_with_fallow_change"] == ["B"]
    assert "fallow_years" not in table["year"].astype(str).values


def test_sentinel_replay_rules_and_explained_regression(mod, tmp_path):
    # two same-date tiles (0.2, 0.4) on 06-01, a sub-floor pair on 06-05 whose mean falls below
    # 0.05 but whose max survives, and a second CSV family whose 06-03 value fills a gap
    (tmp_path / "ndvi_A_no_mask_2015.csv").write_text(
        "A,20150601T1_T1,20150601T1_T2,20150605T1_T1,20150605T1_T2\n,0.2,0.4,0.01,0.08\n"
    )
    (tmp_path / "ndvi_A_sentinel_no_mask_2015.csv").write_text("A,20150603T1_T1\n,0.3\n")
    mean = mod.replay_sentinel_ndvi(str(tmp_path), DATES, ["A"], "mean")
    mx = mod.replay_sentinel_ndvi(str(tmp_path), DATES, ["A"], "max")
    assert mean[0, 0] == pytest.approx(0.3) and mx[0, 0] == pytest.approx(0.4)
    assert mean[2, 0] == pytest.approx(0.3) and mx[2, 0] == pytest.approx(0.3)
    assert np.isnan(mean[4, 0]) and mx[4, 0] == pytest.approx(0.08)

    ss = np.full((len(DATES), 1), 0.5)
    new = _container(["A"], ss, ss, [{}], [{}])
    old = _container(["A"], ss, ss, [{}], [{}])
    new.create_array("remote_sensing/ndvi/sentinel/no_mask", data=mean)
    old.create_array("remote_sensing/ndvi/sentinel/no_mask", data=mx)
    new.create_array("remote_sensing/ndvi/landsat/no_mask", data=mean)
    old.create_array("remote_sensing/ndvi/landsat/no_mask", data=mean)
    reg = mod.check_regression(new, old)
    assert not reg["arrays"]["remote_sensing/ndvi/sentinel/no_mask"]["identical"]
    assert reg["pass"]  # sentinel is a declared, separately verified exception
    attribution = mod.explain_sentinel_difference(new, old, str(tmp_path))
    assert attribution["pass"] and attribution["cells_differing"] == 1
    assert attribution["cells_valid_baseline_only"] == 1
    assert attribution["csv_families"] == {"no_mask": 1, "sentinel_no_mask": 1}
    # an undeclared array that differs still fails
    old.create_array("properties/soils/awc", data=np.array([0.1], np.float32))
    new.create_array("properties/soils/awc", data=np.array([0.2], np.float32))
    assert not mod.check_regression(new, old)["pass"]
