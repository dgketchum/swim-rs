"""Unit tests for the E2 reference-ET sidecar exporter.

Covers the pure pieces of ``examples/6_Flux_International/espa/export_refet_ratio.py``:
manifest construction from a local vector file, UTC-offset rounding, deterministic
selectors, the one-Daily-object-per-day contract for ETo/ETr, and the dry-run path
never starting an Earth Engine task.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ESPA_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "espa"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(ESPA_DIR))
    spec = importlib.util.spec_from_file_location(
        "export_refet_ratio", ESPA_DIR / "export_refet_ratio.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(ESPA_DIR)]
    return module


def _write_cohort(path, records):
    # GeoSeries.from_xy avoids a shapely/geopandas type mismatch that a plain list of
    # shapely objects triggers in this environment.
    centroids = gpd.GeoSeries.from_xy(
        [r[1] for r in records], [r[2] for r in records], crs="EPSG:4326"
    )
    gdf = gpd.GeoDataFrame({"sid": [r[0] for r in records]}, geometry=centroids.buffer(0.0015))
    gdf.to_file(path, driver="FlatGeobuf", engine="fiona")
    return str(path)


@pytest.mark.parametrize(
    "lon, expected",
    [(-121.5, -8), (-96.5, -6), (0.4, 0), (7.6, 1), (-7.6, -1), (150.0, 10), (-97.5, -6)],
)
def test_local_utc_offset_matches_round_lon_over_15(mod, lon, expected):
    assert mod.local_utc_offset(lon) == expected


def test_year_selectors_are_id_then_eto_etr_pairs_in_date_order(mod):
    days = mod.days_in_year(2016)
    assert len(days) == 366
    sel = mod.year_selectors("sid", days[:2])
    assert sel == ["sid", "eto_20160101", "etr_20160101", "eto_20160102", "etr_20160102"]


def test_daily_refet_bands_come_from_one_daily_object(mod):
    daily_cls = MagicMock()
    daily = daily_cls.era5_land.return_value
    hourly = object()

    eto_band, etr_band = mod.daily_refet_bands(hourly, "20150601", daily_cls)

    daily_cls.era5_land.assert_called_once_with(hourly)
    daily.eto.rename.assert_called_once_with("eto_20150601")
    daily.etr.rename.assert_called_once_with("etr_20150601")
    assert eto_band is daily.eto.rename.return_value
    assert etr_band is daily.etr.rename.return_value


def test_build_year_image_requests_every_local_day_once(mod, monkeypatch):
    fake_ee = MagicMock()
    monkeypatch.setattr(mod, "ee", fake_ee)
    monkeypatch.setattr(mod, "_local_day_utc_bounds", lambda d, off: (f"s{d}", f"e{d}"))
    daily_cls = MagicMock()
    hourly = MagicMock()

    mod.build_year_image(hourly, 2015, -8, daily_cls)

    assert hourly.filterDate.call_count == 365
    assert daily_cls.era5_land.call_count == 365
    bands = fake_ee.Image.call_args[0][0]
    assert len(bands) == 730


def test_request_manifest_groups_sites_by_offset(mod, tmp_path):
    path = _write_cohort(
        tmp_path / "cohort.fgb",
        [
            ("US-Bi1", -121.5, 38.1),
            ("US-Tw3", -121.6, 38.1),
            ("US-Ne1", -96.5, 41.2),
            ("DE-RuS", 6.4, 50.9),
        ],
    )
    m = mod.build_request_manifest(path, [2015, 2016], feature_id="sid")

    assert set(m["utc_offset"]) == {-8, -6, 0}
    assert len(m) == 6
    row = m[(m["year"] == 2015) & (m["utc_offset"] == -8)].iloc[0]
    assert row["sites"] == "US-Bi1;US-Tw3"
    assert row["n_sites"] == 2
    assert row["n_days"] == 365
    assert row["description"] == "refet_ratio_2015_utc_m08"
    assert m[m["year"] == 2016]["n_days"].eq(366).all()
    # every site appears in exactly one offset group
    all_sites = [s for row in m[m["year"] == 2015]["sites"] for s in row.split(";")]
    assert sorted(all_sites) == ["DE-RuS", "US-Bi1", "US-Ne1", "US-Tw3"]


def test_request_manifest_rejects_duplicate_ids(mod, tmp_path):
    path = _write_cohort(tmp_path / "dup.fgb", [("US-Bi1", -121.5, 38.1), ("US-Bi1", -121.6, 38.2)])
    with pytest.raises(ValueError, match="duplicate sid"):
        mod.build_request_manifest(path, [2015])


def test_select_requests_pilot_and_check_dir(mod, tmp_path):
    m = pd.DataFrame(
        {
            "year": [2015, 2015, 2016],
            "utc_offset": [-8, -6, -8],
            "suffix": ["utc_m08", "utc_m06", "utc_m08"],
            "description": [
                "refet_ratio_2015_utc_m08",
                "refet_ratio_2015_utc_m06",
                "refet_ratio_2016_utc_m08",
            ],
            "n_sites": [2, 1, 2],
            "n_days": [365, 365, 366],
            "sites": ["a;b", "c", "a;b"],
        }
    )
    pilot = mod.select_requests(m, pilot=(2015, "utc_m08"))
    assert pilot["description"].tolist() == ["refet_ratio_2015_utc_m08"]

    (tmp_path / "refet_ratio_2015_utc_m06.csv").write_text("sid\n")
    rest = mod.select_requests(m, check_dir=str(tmp_path))
    assert rest["description"].tolist() == ["refet_ratio_2015_utc_m08", "refet_ratio_2016_utc_m08"]

    with pytest.raises(ValueError, match="not in the request manifest"):
        mod.select_requests(m, pilot=(2014, "utc_m08"))


def test_dry_run_starts_no_task(mod, monkeypatch):
    fake_ee = MagicMock()
    monkeypatch.setattr(mod, "ee", fake_ee)
    requests = pd.DataFrame(
        {
            "year": [2015],
            "utc_offset": [-8],
            "suffix": ["utc_m08"],
            "description": ["refet_ratio_2015_utc_m08"],
            "n_sites": [2],
            "n_days": [365],
            "sites": ["a;b"],
        }
    )
    ledger = mod.start_exports(
        requests, None, "sid", "wudr", "prefix", dry_run=True, daily_cls=MagicMock()
    )

    fake_ee.batch.Export.table.toCloudStorage.assert_not_called()
    assert ledger == [
        {
            "description": "refet_ratio_2015_utc_m08",
            "year": 2015,
            "utc_offset": -8,
            "n_sites": 2,
            "gcs_uri": "gs://wudr/prefix/refet_ratio_2015_utc_m08.csv",
            "dry_run": True,
            "task_id": None,
        }
    ]


def test_live_export_uses_offset_filter_scale_and_selectors(mod, monkeypatch):
    fake_ee = MagicMock()
    monkeypatch.setattr(mod, "ee", fake_ee)
    monkeypatch.setattr(mod, "build_year_image", lambda *a, **k: MagicMock())
    fc = MagicMock()
    requests = pd.DataFrame(
        {
            "year": [2016],
            "utc_offset": [-6],
            "suffix": ["utc_m06"],
            "description": ["refet_ratio_2016_utc_m06"],
            "n_sites": [1],
            "n_days": [366],
            "sites": ["c"],
        }
    )
    task = fake_ee.batch.Export.table.toCloudStorage.return_value
    task.id = "TASK1"

    ledger = mod.start_exports(
        requests, fc, "sid", "wudr", "prefix", dry_run=False, daily_cls=MagicMock()
    )

    fake_ee.Filter.eq.assert_called_once_with("utc_offset_hours", -6)
    kwargs = fake_ee.batch.Export.table.toCloudStorage.call_args.kwargs
    assert kwargs["fileNamePrefix"] == "prefix/refet_ratio_2016_utc_m06"
    assert len(kwargs["selectors"]) == 1 + 2 * 366
    task.start.assert_called_once()
    assert ledger[0]["task_id"] == "TASK1"
