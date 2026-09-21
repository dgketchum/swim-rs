"""Unit tests for the E2 Gate G6 delivery verification
(``examples/6_Flux_International/container_build/e2_refooting/phase6_verify_delivery.py``).

Covers the pure pieces: product-ID parsing, expected EPSG from the payload projection block,
MTL parsing (first occurrence wins), chip-bounds tolerance, terminal categories and identity
problems, the best-outcome ranking for a site-date served by several scenes (keyed by
(site, product_id) because neighbouring sites share scenes), and the coverage tables.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.coords import BoundingBox

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "container_build" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase6_verify_delivery", E2_DIR / "phase6_verify_delivery.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


def test_parse_product_id_and_epsg(mod):
    p = mod.parse_product_id("LE07_L2SP_022032_20170308_20200831_02_T1")
    assert p == {"sensor": "LE07", "pathrow": "022032", "date": "20170308"}
    with pytest.raises(ValueError):
        mod.parse_product_id("LE07_022032_20170308")
    assert mod.expected_epsg({"projection": {"utm": {"zone": 16, "zone_ns": "north"}}}) == 32616
    assert mod.expected_epsg({"projection": {"utm": {"zone": 55, "zone_ns": "south"}}}) == 32755


def test_read_mtl_keeps_first_occurrence(mod, tmp_path):
    mtl = tmp_path / "x_MTL.txt"
    mtl.write_text(
        'GROUP = A\n    LANDSAT_PRODUCT_ID = "LE07_L2SP_022032_20170308_20200831_02_T1"\n'
        '    SPACECRAFT_ID = "LANDSAT_7"\n    DATE_ACQUIRED = 2017-03-08\n'
        '    LANDSAT_PRODUCT_ID = "LE07_L1TP_022032_20170308_20200831_02_T1"\n'
    )
    m = mod.read_mtl(str(mtl))
    assert m["LANDSAT_PRODUCT_ID"] == "LE07_L2SP_022032_20170308_20200831_02_T1"
    assert m["SPACECRAFT_ID"] == "LANDSAT_7" and m["DATE_ACQUIRED"] == "2017-03-08"


def test_bounds_ok_tolerates_grid_snap_only(mod):
    ext = {"west": 0.0, "east": 4000.0, "south": 0.0, "north": 4000.0}
    assert mod.bounds_ok(BoundingBox(0.0, 10.0, 3990.0, 4000.0), ext)  # ESPA 30 m snap
    assert not mod.bounds_ok(BoundingBox(0.0, 0.0, 2000.0, 4000.0), ext)  # half chip
    assert not mod.bounds_ok(BoundingBox(-100.0, 0.0, 3900.0, 4000.0), ext)  # shifted


def _rec(**kw):
    base = {
        "requested": True,
        "tarball_exists": True,
        "etf_exists": True,
        "md5_ok": True,
        "site_pixel_count": 40,
        "site_mean_etf": 0.5,
        "product_id": "LE07_L2SP_022032_20170308_20200831_02_T1",
        "sensor": "LE07",
        "date": "20170308",
        "mtl_product_id": "LE07_L2SP_022032_20170308_20200831_02_T1",
        "mtl_spacecraft": "LANDSAT_7",
        "mtl_date_acquired": "2017-03-08",
        "epsg": 32616,
        "epsg_expected": 32616,
        "nodata": -9999.0,
        "bounds_ok": True,
    }
    base.update(kw)
    return base


def test_categorize_terminal_categories(mod):
    assert mod.categorize(_rec()) == "delivered-valid-file"
    assert mod.categorize(_rec(site_pixel_count=0, site_mean_etf=np.nan)) == "delivered-nodata"
    assert mod.categorize(_rec(site_mean_etf=3.0)) == "delivered-implausible"
    assert mod.categorize(_rec(md5_ok=False)) == "checksum-fail"
    assert mod.categorize(_rec(etf_exists=False)) == "missing-file"
    assert mod.categorize(_rec(requested=False)) == "manifest-error"
    # a valid file below the ingest floor is still a valid delivery
    assert mod.categorize(_rec(site_mean_etf=0.01)) == "delivered-valid-file"


def test_identity_ok_reports_each_problem(mod):
    assert mod.identity_ok(_rec(), "LE07") == (True, "")
    ok, problems = mod.identity_ok(
        _rec(mtl_spacecraft="LANDSAT_8", epsg=32615, nodata=0.0, bounds_ok=False), "LC08"
    )
    assert not ok
    assert set(problems.split(";")) == {"spacecraft", "manifest_sensor", "epsg", "nodata", "bounds"}
    ok, problems = mod.identity_ok(_rec(mtl_date_acquired="2017-03-09"), None)
    assert not ok and problems == "date"


def _ledger(rows):
    cols = ["site", "product_id", "requested", "category", "site_mean_etf"]
    return pd.DataFrame(rows, columns=cols)


def _scenes(rows):
    cols = ["site", "date", "year", "sensor", "product_id", "request", "status"]
    return pd.DataFrame(rows, columns=cols)


def test_site_date_outcomes_ranks_best_scene_per_site_date(mod):
    shared = "LE07_L2SP_022032_20170308_20200831_02_T1"
    ledger = _ledger(
        [
            # same scene serves two sites with different results
            ["A", shared, True, "delivered-valid-file", 0.5],
            ["B", shared, True, "delivered-nodata", np.nan],
            # site A alternate row on the same date: nodata, must lose to the valid one
            ["A", "LE07_L2SP_022033_20170308_20200831_02_T1", True, "delivered-nodata", np.nan],
            # valid but below the ingest floor
            ["A", "LE07_L2SP_022032_20170324_20200831_02_T1", True, "delivered-valid-file", 0.01],
        ]
    )
    scenes = _scenes(
        [
            ["A", "20170308", "2017", "LE07", shared, "True", "never_ordered"],
            [
                "A",
                "20170308",
                "2017",
                "LE07",
                "LE07_L2SP_022033_20170308_20200831_02_T1",
                "True",
                "never_ordered",
            ],
            ["B", "20170308", "2017", "LE07", shared, "True", "never_ordered"],
            [
                "A",
                "20170324",
                "2017",
                "LE07",
                "LE07_L2SP_022032_20170324_20200831_02_T1",
                "True",
                "never_ordered",
            ],
            [
                "A",
                "20180101",
                "2018",
                "LE07",
                "LE07_L2SP_022032_20180101_20200831_02_T1",
                "False",
                "delivered_not_ingested",
            ],
            [
                "B",
                "20180101",
                "2018",
                "LC08",
                "LC08_L2SP_022032_20180101_20200831_02_T1",
                "False",
                "native_below_min_etf",
            ],
        ]
    )
    out = mod.site_date_outcomes(ledger, scenes).set_index(["site", "date"])["outcome"]
    assert out[("A", "20170308")] == "new_valid"
    assert out[("B", "20170308")] == "new_nodata"
    assert out[("A", "20170324")] == "new_valid_below_min_etf"
    assert out[("A", "20180101")] == "recoverable_delivered_not_ingested"
    assert out[("B", "20180101")] == "native_below_min_etf"
    assert len(out) == 5

    # a duplicated (site, product_id) ledger row is a hard error
    dup = pd.concat([ledger, ledger.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        mod.site_date_outcomes(dup, scenes)


def test_coverage_tables_count_pairs_before_and_after(mod):
    table = pd.DataFrame(
        {
            "site": ["A", "A", "A", "B", "B"],
            "date": ["20170308", "20170324", "20170409", "20170308", "20180101"],
            "ssebop_container": [np.nan, np.nan, 0.6, np.nan, 0.4],
            "ptjpl": [0.5, 0.4, 0.7, 0.3, np.nan],
            "eto": [3.0, 3.0, 4.0, 3.0, 1.0],
            "year": [2017, 2017, 2017, 2017, 2018],
        }
    )
    outcomes = pd.DataFrame(
        {
            "site": ["A", "A", "B"],
            "date": ["20170308", "20170324", "20170308"],
            "year": ["2017", "2017", "2017"],
            "sensor": ["LE07", "LE07", "LE07"],
            "outcome": ["new_valid", "recoverable_delivered_not_ingested", "new_nodata"],
        }
    )
    by_year, by_site = mod.coverage_tables(table, outcomes)
    y17 = by_year.set_index("year").loc[2017]
    assert y17["n_ptjpl_dates"] == 4 and y17["paired_now"] == 1
    assert y17["paired_after_delivery"] == 2 and y17["paired_after_recovery"] == 3
    assert y17["residual_new_nodata"] == 1
    y18 = by_year.set_index("year").loc[2018]
    assert y18["n_ptjpl_dates"] == 0 and y18["n_ssebop_only_now"] == 1
    a = by_site.set_index("site").loc["A"]
    assert a["frac_paired_after_recovery"] == pytest.approx(1.0)
