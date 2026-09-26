"""Tests for the Example 7 path helpers (``examples/7_Applied_Water/ex7_paths.py``)."""

import importlib.util
from pathlib import Path

import pytest

EX7 = Path(__file__).resolve().parents[2] / "examples" / "7_Applied_Water"


@pytest.fixture(scope="module")
def ex7_paths():
    spec = importlib.util.spec_from_file_location("ex7_paths", EX7 / "ex7_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_toml_resolves_under_root_and_project(ex7_paths):
    root = ex7_paths.swim_root()
    ws = ex7_paths.project_ws()
    assert ws == root / "7_Applied_Water"
    assert ex7_paths.data_dir() == ws / "data"
    assert ex7_paths.gis_dir() == ws / "data" / "gis"
    assert ex7_paths.fields_shp().name == "applied_water_fields.shp"
    assert ex7_paths.base_container() == ws / "data" / "7_Applied_Water.swim"
    assert ex7_paths.run_container() == ws / "data" / "7_Applied_Water_e7cal.swim"
    assert ex7_paths.pest_run_dir() == ws / "pestrun"
    assert ex7_paths.archive_dir() == ws / "results" / "e7cal" / "archive"
    assert ex7_paths.merged_posterior().name == "merged_posterior.json"
    assert ex7_paths.eval_dir("calibrated") == ws / "results" / "applied_calibrated"
    assert ex7_paths.TRUTH_CSV == EX7 / "data" / "metered_truth.csv"


def test_root_relocates_every_workspace_path(ex7_paths, tmp_path):
    toml = tmp_path / "relocated.toml"
    toml.write_text(
        'project = "7_Applied_Water"\n'
        'root = "/elsewhere/swim"\n'
        "[paths]\n"
        'project_workspace = "{root}/{project}"\n'
        'data = "{project_workspace}/data"\n'
        'gis = "{data}/gis"\n'
        'fields_shapefile = "{gis}/applied_water_fields.shp"\n'
        'container = "{data}/{project}.swim"\n'
        "[calibration]\n"
        'pest_run_dir = "{project_workspace}/pestrun"\n'
    )
    ws = Path("/elsewhere/swim/7_Applied_Water")
    assert ex7_paths.project_ws(toml) == ws
    assert ex7_paths.fields_shp(toml) == ws / "data" / "gis" / "applied_water_fields.shp"
    assert ex7_paths.run_container("other", toml) == ws / "data" / "7_Applied_Water_other.swim"
    assert ex7_paths.pest_run_dir(toml) == ws / "pestrun"
    assert ex7_paths.results_root(toml) == ws / "results"
    assert not ws.exists()


def test_excluded_field_years_are_dropped(ex7_paths):
    import pandas as pd

    assert set(ex7_paths.EXCLUDED_FIELD_YEARS) == {
        ("SLV_013", 2019),
        ("SLV_024", 2019),
        ("SLV_025", 2019),
        ("SLV_034", 2019),
    }
    df = pd.DataFrame(
        {
            "site_id": ["SLV_013", "SLV_013", "SLV_018", "SLV_034"],
            "year": [2019, 2018, 2019, 2019],
            "metered_depth_mm": [684.6, 700.0, 650.0, 807.1],
        }
    )
    kept = ex7_paths.drop_excluded_field_years(df)
    assert list(zip(kept.site_id, kept.year)) == [("SLV_013", 2018), ("SLV_018", 2019)]
    assert len(df) == 4
