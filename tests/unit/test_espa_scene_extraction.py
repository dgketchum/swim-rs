"""Unit tests for scene-identity preservation in the ESPA ETF extraction and CSV writing.

Covers ``examples/6_Flux_International/container_build/espa/espa_extract_etf.py`` (product-ID parsing and
ordering, legacy-schema detection) and ``espa_write_etf_csvs.py`` (legacy ``ETF_`` columns vs
scene-key columns, duplicate scene key is a hard error), and checks that the container ingestor
collapses two same-date scene-key columns deterministically (landsat: max), so same-overpass
alternates are never silently overwritten upstream.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swimrs.container.components.ingestor import _parse_single_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
ESPA_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "container_build" / "espa"


def _load(name):
    sys.path.insert(0, str(ESPA_DIR))
    spec = importlib.util.spec_from_file_location(name, ESPA_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(ESPA_DIR)]
    return module


@pytest.fixture(scope="module")
def extract():
    return _load("espa_extract_etf")


@pytest.fixture(scope="module")
def writer():
    return _load("espa_write_etf_csvs")


# --------------------------------------------------------------------------- extractor
def test_etf_regex_parses_full_product_id(extract):
    m = extract.ETF_RE.match("LE07_L2SP_112082_20130115_20200908_02_T1_ETF.tif")
    assert m.group("product_id") == "LE07_L2SP_112082_20130115_20200908_02_T1"
    assert m.group("sensor") == "LE07"
    assert m.group("pathrow") == "112082"
    assert m.group("date") == "20130115"
    assert extract.ETF_RE.match("LE07_L2SP_112082_20130115_20200908_02_T1_ETA.tif") is None
    assert extract.ETF_RE.match("LE07_112082_20130115_ETF.tif") is None


def test_find_etf_tifs_orders_by_date_then_product_id(extract, tmp_path):
    names = [
        "LE07_L2SP_112083_20130115_20200908_02_T1_ETF.tif",
        "LE07_L2SP_112082_20130115_20200908_02_T1_ETF.tif",
        "LC08_L2SP_112082_20130107_20200908_02_T1_ETF.tif",
        "LC08_L2SP_112082_20130107_20200908_02_T1_ETA.tif",
        "README.txt",
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    found = extract._find_etf_tifs(tmp_path)
    assert [g["product_id"] for _, g in found] == [
        "LC08_L2SP_112082_20130107_20200908_02_T1",
        "LE07_L2SP_112082_20130115_20200908_02_T1",
        "LE07_L2SP_112083_20130115_20200908_02_T1",
    ]
    assert extract._find_etf_tifs(tmp_path / "missing") == []


def test_extract_all_refuses_sites_absent_from_shapefile(extract, tmp_path, monkeypatch):
    manifest = tmp_path / "espa_manifest.csv"
    pd.DataFrame(
        {
            "site": ["US-Ne1", "US-Ne2", "AU-RDF"],
            "year": ["2014", "2014", "2015"],
            "download_status": ["complete", "pending", "complete"],
            "output_dir": [str(tmp_path / s) for s in ("a", "b", "c")],
            "extract_status": ["", "", ""],
        }
    ).to_csv(manifest, index=False)

    class FakeSites:
        index = pd.Index(["US-Ne1", "US-Ne2"])

    monkeypatch.setattr(extract, "_load_site_geometries", lambda shp: FakeSites())
    calls = []
    monkeypatch.setattr(extract, "extract_site_year", lambda *a: calls.append(a) or {})

    with pytest.raises(ValueError, match=r"1 downloaded site\(s\) absent .*AU-RDF"):
        extract.extract_all(manifest, Path("cohort.shp"))

    assert calls == [], "no extraction may run when a downloaded site lacks a geometry"
    assert list((tmp_path / "extracts" / "etf_json").glob("*.json")) == []
    # Undownloaded rows never need a geometry
    assert (pd.read_csv(manifest, dtype=str)["extract_status"].fillna("") == "").all()


def test_is_legacy_date_keyed(extract):
    assert extract.is_legacy_date_keyed({"2013-01-15": {"mean": 0.5}})
    assert not extract.is_legacy_date_keyed(
        {"LE07_L2SP_112082_20130115_20200908_02_T1": {"date": "2013-01-15"}}
    )
    assert not extract.is_legacy_date_keyed({})


# --------------------------------------------------------------------------- writer
def test_site_row_legacy_schema_writes_etf_columns(writer):
    site_data = {
        "2013-01-15": {"mean": 0.512345678, "count": 10},
        "2013-01-07": {"mean": None, "count": 0},
        "2013-02-16": {"mean": 0.7, "count": 10},
    }
    row = writer.site_row("S", site_data)
    assert row == {"sid": "S", "ETF_20130115": 0.512346, "ETF_20130216": 0.7}


def test_site_row_scene_schema_keeps_same_date_alternates(writer):
    site_data = {
        "LE07_L2SP_112083_20130115_20200908_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112083",
            "mean": 0.4,
        },
        "LE07_L2SP_112082_20130115_20200908_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112082",
            "mean": 0.6,
        },
        "LC08_L2SP_112082_20130107_20200908_02_T1": {
            "date": "2013-01-07",
            "sensor": "LC08",
            "pathrow": "112082",
            "mean": None,
        },
        "LC08_L2SP_112082_20130123_20200908_02_T1": {
            "date": "2013-01-23",
            "sensor": "LC08",
            "pathrow": "112082",
            "mean": 0.55,
        },
    }
    row = writer.site_row("S", site_data)
    # both alternates survive as separate columns, ordered by (date, product_id); NaN mean skipped
    assert list(row) == [
        "sid",
        "LE07_112082_20130115",
        "LE07_112083_20130115",
        "LC08_112082_20130123",
    ]
    assert row["LE07_112082_20130115"] == 0.6 and row["LE07_112083_20130115"] == 0.4


def test_site_row_duplicate_scene_key_is_hard_error(writer):
    site_data = {
        "LE07_L2SP_112082_20130115_20200908_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112082",
            "mean": 0.4,
        },
        "LE07_L2SP_112082_20130115_20210101_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112082",
            "mean": 0.6,
        },
    }
    with pytest.raises(ValueError, match="delivered twice"):
        writer.site_row("S", site_data)


# --------------------------------------------------------------------------- ingestor
def test_ingestor_collapses_same_date_scene_keys_by_max(writer, tmp_path):
    site_data = {
        "LE07_L2SP_112083_20130115_20200908_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112083",
            "mean": 0.4,
        },
        "LE07_L2SP_112082_20130115_20200908_02_T1": {
            "date": "2013-01-15",
            "sensor": "LE07",
            "pathrow": "112082",
            "mean": 0.6,
        },
        "LC08_L2SP_112082_20130123_20200908_02_T1": {
            "date": "2013-01-23",
            "sensor": "LC08",
            "pathrow": "112082",
            "mean": 0.55,
        },
    }
    row = writer.site_row("S", site_data)
    csv = tmp_path / "ssebop_etf_S_no_mask_2013.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    series = _parse_single_csv(csv, "sid", "landsat", {"S"}, None)
    assert len(series) == 1
    s = series[0]
    assert s.name == "S"
    assert list(s.index) == [pd.Timestamp("2013-01-15"), pd.Timestamp("2013-01-23")]
    assert s.loc[pd.Timestamp("2013-01-15")] == pytest.approx(0.6)  # max of 0.6 and 0.4
    assert s.loc[pd.Timestamp("2013-01-23")] == pytest.approx(0.55)
    # legacy ETF_ columns parse to the same dates
    legacy = pd.DataFrame([{"sid": "S", "ETF_20130115": 0.6, "ETF_20130123": np.nan}])
    legacy.to_csv(csv, index=False)
    s2 = _parse_single_csv(csv, "sid", "landsat", {"S"}, None)[0]
    assert s2.loc[pd.Timestamp("2013-01-15")] == pytest.approx(0.6)
    assert np.isnan(s2.loc[pd.Timestamp("2013-01-23")])
