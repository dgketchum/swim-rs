"""Unit tests for the Example 8 (SCAN soil-moisture) new logic.

Covers the two modules with genuinely new code (the other Ex8 scripts are thin
re-dispatch wrappers over the already-tested Example 5 pipeline):

- build_scan_fields.py: SCAN UID sanitization, 150 m buffer geometry, and the
  DBF column-collision regression (fields shapefile must NOT carry lowercase
  lat/lon/elev, which collide with the uppercase LAT/LON/ELEV that
  assign_gridmet_ids adds, silently breaking download_gridmet).
- evaluate.py: the theta_available formula and day-of-year deseasonalization.

The example scripts live outside the swimrs package, so they are imported by
path via importlib. build_scan_fields reads the checked-in scan_shortlist.csv;
if that file is absent the shapefile-shape tests skip (the pure-function tests
still run).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EX8 = Path(__file__).resolve().parents[2] / "examples" / "8_Soil_Moisture"


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(f"ex8_{module_name}", EX8 / f"{module_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# build_scan_fields.py
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bsf():
    return _load("build_scan_fields")


@pytest.mark.parametrize(
    "uid, expected",
    [
        ("SCAN:Lind#1", "Lind1"),  # drops network prefix + '#'
        ("SCAN:Circleville", "Circleville"),
        ("SCAN:Table Mountain", "TableMountain"),  # drops whitespace
        ("SCAN:UAPB-Earle", "UAPBEarle"),  # drops hyphen
        ("PlainName", "PlainName"),  # no prefix -> unchanged
    ],
)
def test_sanitize(bsf, uid, expected):
    assert bsf._sanitize(uid) == expected


def test_build_shapefile_has_no_dbf_colliding_columns(bsf):
    """Regression: lowercase lat/lon/elev collide with assign_gridmet_ids' LAT/LON/ELEV."""
    if not bsf.SHORTLIST.exists():
        pytest.skip("scan_shortlist.csv not present")
    fields, sites = bsf.build()

    cols = set(fields.columns)
    # the columns that would case-collide in a DBF must be absent...
    assert "lat" not in cols and "lon" not in cols and "elev" not in cols
    # ...and the safely-renamed coordinate columns must be present
    assert {"site_lat", "site_lon", "site_id", "geometry"} <= cols
    # feature ids are unique and DBF/filename-safe (alphanumeric)
    assert fields["site_id"].is_unique
    assert fields["site_id"].str.match(r"^[0-9A-Za-z]+$").all()


def test_build_buffer_geometry(bsf):
    if not bsf.SHORTLIST.exists():
        pytest.skip("scan_shortlist.csv not present")
    fields, _ = bsf.build(buffer_m=150.0)

    assert str(fields.crs).endswith("5071")  # equal-area for metric buffering
    # 150 m radius circle -> pi*r^2 ~= 70,686 m^2; allow generous tolerance
    areas = fields.geometry.area
    assert np.allclose(areas, np.pi * 150.0**2, rtol=0.02)
    assert (fields.geometry.geom_type == "Polygon").all()


def test_build_sites_join_table(bsf):
    if not bsf.SHORTLIST.exists():
        pytest.skip("scan_shortlist.csv not present")
    fields, sites = bsf.build()

    assert list(sites["site_id"]) == list(fields["site_id"])
    # theta_csv points at the SCAN parquet archive, keyed by the un-prefixed UID
    assert sites["theta_csv"].str.endswith(".parquet").all()
    assert sites["theta_csv"].str.contains("/soil_moisture/scan/SCAN_").all()


# --------------------------------------------------------------------------- #
# evaluate.py
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ev():
    return _load("evaluate")


def test_theta_available_scalar(ev):
    # awc=200 mm/m, zr=1.0 m -> 200 mm capacity; depl=50 mm; daw3=10 mm; zr_max=2.0 m
    # soil_water = 200*1 - 50 + 10 = 160 mm; /(2*1000) = 0.08 m3/m3
    assert ev.theta_available(200.0, 1.0, 50.0, 10.0, 2.0) == pytest.approx(0.08)


def test_theta_available_vectorized(ev):
    awc, zr_max = 180.0, 1.5
    zr = np.array([0.5, 1.0, 1.5])
    depl = np.array([0.0, 30.0, 90.0])
    daw3 = np.array([0.0, 5.0, 0.0])
    got = ev.theta_available(awc, zr, depl, daw3, zr_max)
    exp = (awc * zr - depl + daw3) / (zr_max * 1000.0)
    assert np.allclose(got, exp)


def test_theta_available_monotonic_in_depletion(ev):
    """More depletion -> less available water (physical sanity)."""
    base = dict(awc=200.0, zr=1.0, daw3=0.0, zr_max=2.0)
    wet = ev.theta_available(depl_root=10.0, **base)
    dry = ev.theta_available(depl_root=120.0, **base)
    assert wet > dry


def test_deseasonalize_removes_doy_climatology(ev):
    # two identical years of a pure seasonal cycle -> anomalies collapse to ~0
    idx = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    doy = idx.dayofyear
    seasonal = pd.Series(np.sin(2 * np.pi * doy / 365.0), index=idx)
    anom = ev._deseasonalize(seasonal)
    assert np.nanmax(np.abs(anom.values)) < 1e-9


def test_deseasonalize_preserves_interannual_anomaly(ev):
    idx = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    s = pd.Series(1.0, index=idx)
    s[s.index.year == 2002] = 3.0  # year 2 offset by +2
    anom = ev._deseasonalize(s)
    # each day's climatology is the 2-year mean (2.0); anomalies are -1 and +1
    assert anom[anom.index.year == 2001].round(6).eq(-1.0).all()
    assert anom[anom.index.year == 2002].round(6).eq(1.0).all()
