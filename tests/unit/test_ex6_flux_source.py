"""Regression test for Example 6 flux-source selection (review finding A4).

`load_flux_et` must honor the cohort shapefile's per-site (network, et_col)
rather than the first file found in the fixed network search order. The
US-Ne1/2/3 sites have an ameriflux soil-moisture-only file (no ET column) that
shadowed their fluxnet ET record, silently dropping them from the cohort.
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EX6 = os.path.join(HERE, "..", "..", "examples", "6_Flux_International")


@pytest.fixture
def ev(monkeypatch):
    """Import the Example 6 evaluate module with QAQC_ROOT pointed at a tmp tree."""
    if EX6 not in sys.path:
        sys.path.insert(0, EX6)
    spec = importlib.util.spec_from_file_location("ex6_evaluate", os.path.join(EX6, "evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_qaqc(root, network, site, columns):
    d = os.path.join(root, network)
    os.makedirs(d, exist_ok=True)
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({c: range(5) for c in columns}, index=idx)
    df.index.name = "date"
    df.to_csv(os.path.join(d, f"{site}_daily_data.csv"))


def test_declared_source_beats_search_order(ev, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(ev, "QAQC_ROOT", root)

    site = "US-Ne1"
    # ameriflux exists first in search order but has NO ET column
    _write_qaqc(root, "ameriflux", site, ["SWC_1", "SWC_2"])
    # fluxnet has the real ET_corr record
    _write_qaqc(root, "fluxnet", site, ["ET_corr", "ET"])

    # Without a declared source: ameriflux wins, no ET column -> empty
    assert ev.load_flux_et(site).empty

    # With the shapefile-declared (network, et_col): the fluxnet record is used
    series = ev.load_flux_et(site, ("fluxnet", "ET_corr"))
    assert not series.empty
    assert series.attrs["et_col"] == "ET_corr"
    assert series.attrs["flux_file"].endswith("fluxnet/US-Ne1_daily_data.csv")


def test_find_flux_file_prefers_declared_network(ev, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(ev, "QAQC_ROOT", root)
    site = "US-Ne2"
    _write_qaqc(root, "ameriflux", site, ["SWC_1"])
    _write_qaqc(root, "fluxnet", site, ["ET_corr"])

    assert ev.find_flux_file(site, "fluxnet").endswith("fluxnet/US-Ne2_daily_data.csv")
    # no network -> first in QAQC_NETWORKS order (ameriflux)
    assert ev.find_flux_file(site).endswith("ameriflux/US-Ne2_daily_data.csv")


def test_load_flux_sources_from_shapefile(ev, tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "sid": ["US-Ne1", "AR-CCa"],
            "network": ["fluxnet", "ameriflux"],
            "et_col": ["ET_corr", "ET"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    shp = tmp_path / "cohort.shp"
    gdf.to_file(shp, engine="fiona")

    sources = ev.load_flux_sources(str(shp), "sid")
    assert sources["US-Ne1"] == ("fluxnet", "ET_corr")
    assert sources["AR-CCa"] == ("ameriflux", "ET")


def test_load_flux_sources_missing_network_column_returns_empty(ev, tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"sid": ["US-Ne1"], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    shp = tmp_path / "cohort_nonet.shp"
    gdf.to_file(shp, engine="fiona")
    assert ev.load_flux_sources(str(shp), "sid") == {}
