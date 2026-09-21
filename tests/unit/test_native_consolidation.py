"""Unit tests for the E2 Phase 7 native-record consolidation
(``examples/6_Flux_International/container_build/e2_refooting/phase7_consolidate_native.py``).

Rules under test: ingested values are preserved exactly, JSON-only dates are added, tree
conflicts resolve to the ingested value (or ``max`` when never ingested), repair scenes become
scene-keyed columns that never replace a legacy value, and duplicates/identity breaks raise.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "container_build" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase7_consolidate_native", E2_DIR / "phase7_consolidate_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


COLS = ["tree", "site", "file_site", "year", "date", "product_id", "mean", "count", "json_path"]
PID = "LE07_L2SP_028031_20150609_20200908_02_T1"
PID_ALT = "LE07_L2SP_028032_20150609_20200908_02_T1"


def _values(rows):
    return pd.DataFrame(
        [
            {
                "tree": t,
                "site": s,
                "file_site": s,
                "year": d[:4],
                "date": d,
                "product_id": p,
                "mean": m,
                "count": 40,
                "json_path": f"{t}.json",
            }
            for t, s, d, p, m in rows
        ],
        columns=COLS,
    )


def _native(rows):
    return pd.DataFrame(rows, columns=["site", "date", "native_value"])


def test_consolidate_rules(mod):
    values = _values(
        [
            ("espa", "A", "20150601", None, 0.5),  # ingested, single tree
            ("espa", "A", "20150617", None, 0.700000),  # ingested, two trees agree
            ("espa_crop99", "A", "20150617", None, 0.7000004),  # equal after rounding
            ("espa", "A", "20150703", None, 0.30),  # conflict; espa value was ingested
            ("espa_crop99", "A", "20150703", None, 0.45),
            ("espa_crop99", "A", "20150719", None, 0.60),  # json-only, conflicting, never ingested
            ("espa_ext_2008_2017", "A", "20150719", None, 0.55),
            ("espa_crop99", "A", "20150804", None, 0.80),  # json-only single tree
            ("espa_le07_repair", "A", "20150609", PID, 0.62),  # scenes, two alternates one date
            ("espa_le07_repair", "A", "20150609", PID_ALT, 0.71),
            ("espa_le07_repair", "A", "20150601", "LC08_L2SP_028031_20150601_20200908_02_T1", 0.9),
            ("espa", "B", "20150601", None, 0.2),  # not in cohort
            ("espa", "A", "20120601", None, 0.2),  # outside years
        ]
    )
    native = _native([("A", "20150601", 0.5), ("A", "20150617", 0.7), ("A", "20150703", 0.3)])
    led = mod.consolidate(values, native, {"A"}, (2013, 2025)).set_index("column")

    assert len(led) == 8
    assert led.loc["ETF_20150703", "value"] == 0.3
    assert led.loc["ETF_20150703", "resolution"] == "kept_ingested_value"
    assert bool(led.loc["ETF_20150703", "conflict"])
    assert led.loc["ETF_20150719", "value"] == 0.6
    assert led.loc["ETF_20150719", "resolution"] == "max_of_trees"
    assert led.loc["ETF_20150617", "n_sources"] == 2
    assert not bool(led.loc["ETF_20150617", "conflict"])
    assert led.loc["ETF_20150804", "kind"] == "legacy_date"
    assert not bool(led.loc["ETF_20150804", "in_native"])
    assert led.loc["LE07_028031_20150609", "value"] == 0.62
    assert led.loc["LE07_028032_20150609", "value"] == 0.71
    assert led.loc["LE07_028031_20150609", "product_id"] == PID
    assert bool(led.loc["LC08_028031_20150601", "same_date_legacy"])
    assert not bool(led.loc["LE07_028031_20150609", "same_date_legacy"])
    assert "B" not in led["site"].values
    assert "ETF_20120601" not in led.index


def test_identity_break_and_missing_source_raise(mod):
    values = _values([("espa_crop99", "A", "20150601", None, 0.55)])
    with pytest.raises(ValueError, match="no tree value matches ingested native"):
        mod.consolidate(values, _native([("A", "20150601", 0.5)]), {"A"}, (2013, 2025))
    # a half-way mean rounded the other way by the legacy writer is the same observation
    halfway = _values([("espa", "A", "20150601", None, 0.4815125)])
    led = mod.consolidate(halfway, _native([("A", "20150601", 0.481513)]), {"A"}, (2013, 2025))
    assert led.loc[0, "value"] == 0.481513 and not bool(led.loc[0, "conflict"])
    with pytest.raises(ValueError, match="no JSON source"):
        mod.consolidate(
            values, _native([("A", "20150601", 0.55), ("A", "20150617", 0.4)]), {"A"}, (2013, 2025)
        )


def test_duplicate_scene_key_and_null_mean_raise(mod):
    dup = _values(
        [
            ("espa_le07_repair", "A", "20150609", PID, 0.62),
            ("espa_le07_repair", "A", "20150609", PID.replace("20200908", "20210101"), 0.63),
        ]
    )
    with pytest.raises(ValueError, match="delivered twice"):
        mod.consolidate(dup, _native([]), {"A"}, (2013, 2025))
    null = _values([("espa", "A", "20150601", None, None)])
    with pytest.raises(ValueError, match="without a mean"):
        mod.consolidate(null, _native([]), {"A"}, (2013, 2025))


def test_write_verify_roundtrip(mod, tmp_path):
    values = _values(
        [
            ("espa", "A", "20150601", None, 0.5),
            ("espa_crop99", "A", "20150804", None, 0.8),
            ("espa_le07_repair", "A", "20150609", PID, 0.62),
            ("espa", "A", "20160601", None, 0.4),
        ]
    )
    native = _native([("A", "20150601", 0.5), ("A", "20160601", 0.4)])
    led = mod.consolidate(values, native, {"A"}, (2013, 2025))
    hashes = mod.write_complete_csvs(led, str(tmp_path))
    assert sorted(Path(p).name for p in hashes) == [
        "ssebop_etf_A_no_mask_2015.csv",
        "ssebop_etf_A_no_mask_2016.csv",
    ]
    wide = pd.read_csv(tmp_path / "ssebop_etf_A_no_mask_2015.csv")
    assert wide.columns.tolist() == ["sid", "ETF_20150601", "LE07_028031_20150609", "ETF_20150804"]
    check = mod.verify_written(led, native, str(tmp_path))
    assert check["pass"] and check["n_columns"] == 4 and check["native_values_checked"] == 2
