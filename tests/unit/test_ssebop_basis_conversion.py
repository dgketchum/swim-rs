"""Unit tests for the E2 SSEBop alfalfa-to-grass basis converter (Phase 2, Gate G2).

Exercises the pure pieces of
``examples/6_Flux_International/container_build/e2_refooting/phase2_convert_ssebop_basis.py``: exact joins,
hand-calculated corrections, the fixed-scalar regression, the missing/invalid statuses, the
no-cap rule, input-hash preservation, and deterministic CSV output.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "container_build" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase2_convert_ssebop_basis", E2_DIR / "phase2_convert_ssebop_basis.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


def _native(rows):
    return pd.DataFrame(rows, columns=["site", "date", "native", "native_csv"])


def _sidecar(rows):
    return pd.DataFrame(rows, columns=["site", "date", "eto", "etr", "source_file"])


def test_ratio_one_returns_native_exactly(mod):
    nat = _native([("A", "20150601", 0.61, "a.csv"), ("A", "20150617", 1.0, "a.csv")])
    side = _sidecar([("A", "20150601", 5.0, 5.0, "s.csv"), ("A", "20150617", 6.5, 6.5, "s.csv")])
    led = mod.convert(nat, side)
    assert (led["status"] == "ok").all()
    assert led["corrected"].tolist() == nat["native"].tolist()


def test_hand_calculated_pairs(mod):
    nat = _native([("A", "20150601", 0.5, "a.csv"), ("B", "20150601", 0.8, "b.csv")])
    side = _sidecar([("A", "20150601", 4.0, 5.0, "s.csv"), ("B", "20150601", 6.0, 7.5, "s.csv")])
    led = mod.convert(nat, side).set_index("site")
    assert led.loc["A", "ratio"] == pytest.approx(1.25)
    assert led.loc["A", "corrected"] == pytest.approx(0.625)
    assert led.loc["B", "corrected"] == pytest.approx(1.0)


def test_varying_ratio_is_not_a_fixed_scalar(mod):
    dates = [f"201506{d:02d}" for d in range(1, 11)]
    nat = _native([("A", d, 0.5, "a.csv") for d in dates])
    side = _sidecar([("A", d, 4.0, 4.0 + 0.1 * i, "s.csv") for i, d in enumerate(dates)])
    led = mod.convert(nat, side)
    assert led["ratio"].nunique() == 10
    scalar_fit = led["corrected"] / led["native"]
    assert scalar_fit.std() > 0  # a fixed-scalar implementation would fail here


def test_missing_sidecar_zero_eto_negative_etr_native_nan_statuses(mod):
    nat = _native(
        [
            ("A", "20150601", 0.5, "a.csv"),
            ("A", "20150602", 0.5, "a.csv"),
            ("A", "20150603", 0.5, "a.csv"),
            ("A", "20150604", np.nan, "a.csv"),
        ]
    )
    side = _sidecar(
        [
            ("A", "20150602", 0.0, 3.0, "s.csv"),
            ("A", "20150603", 3.0, -1.0, "s.csv"),
            ("A", "20150604", 3.0, 4.0, "s.csv"),
        ]
    )
    led = mod.convert(nat, side).set_index("date")
    assert led.loc["20150601", "status"] == "no_sidecar"
    assert led.loc["20150602", "status"] == "nonpositive_eto"
    assert led.loc["20150603", "status"] == "nonpositive_etr"
    assert led.loc["20150604", "status"] == "native_null"
    assert led["corrected"].isna().all()
    assert led["ratio"].isna().all()


def test_duplicate_site_date_is_hard_error(mod):
    nat = _native([("A", "20150601", 0.5, "a.csv"), ("A", "20150601", 0.6, "a2.csv")])
    side = _sidecar([("A", "20150601", 4.0, 5.0, "s.csv")])
    with pytest.raises(ValueError, match="duplicate"):
        mod.convert(nat, side)
    nat = _native([("A", "20150601", 0.5, "a.csv")])
    side = _sidecar([("A", "20150601", 4.0, 5.0, "s.csv"), ("A", "20150601", 4.0, 5.1, "t.csv")])
    with pytest.raises(ValueError, match="duplicate"):
        mod.convert(nat, side)


def test_values_above_one_survive_and_flags_are_set(mod):
    nat = _native([("A", "20150601", 1.0, "a.csv"), ("A", "20151215", 0.4, "a.csv")])
    side = _sidecar([("A", "20150601", 5.0, 6.5, "s.csv"), ("A", "20151215", 0.5, 1.2, "s.csv")])
    led = mod.convert(nat, side).set_index("date")
    assert led.loc["20150601", "corrected"] == pytest.approx(1.3)
    assert not led.loc["20150601", "low_eto"]
    assert not led.loc["20150601", "ratio_outside_screen"]
    assert led.loc["20151215", "low_eto"]
    assert led.loc["20151215", "ratio_outside_screen"]
    assert led.loc["20151215", "status"] == "ok"  # flagged, never filtered


def test_scene_identity_join_and_unknown_default(mod):
    nat = _native([("A", "20150601", 0.5, "a.csv"), ("A", "20150609", 0.5, "a.csv")])
    side = _sidecar([("A", "20150601", 4.0, 5.0, "s.csv"), ("A", "20150609", 4.0, 5.0, "s.csv")])
    scenes = pd.DataFrame(
        {
            "site": ["A"],
            "date": ["20150601"],
            "sensor": ["LE07"],
            "product_ids": ["LE07_L2SP_044033_20150601_20200903_02_T1"],
            "n_products": [1],
            "trees": ["espa"],
        }
    )
    led = mod.convert(nat, side, scenes).set_index("date")
    assert led.loc["20150601", "sensor"] == "LE07"
    assert led.loc["20150609", "sensor"] == "unknown"
    assert led.loc["20150609", "n_products"] == 0


def test_load_native_rejects_mismatched_year_and_duplicates(mod, tmp_path):
    good = tmp_path / "ssebop_etf_A_no_mask_2015.csv"
    good.write_text("sid,ETF_20150601,ETF_20150617\nA,0.5,0.6\n")
    nat = mod.load_native(str(tmp_path), {"A"}, (2013, 2025))
    assert nat["date"].tolist() == ["20150601", "20150617"]
    bad = tmp_path / "ssebop_etf_B_no_mask_2015.csv"
    bad.write_text("sid,ETF_20160601\nB,0.5\n")
    with pytest.raises(ValueError, match="outside file year"):
        mod.load_native(str(tmp_path), {"B"}, (2013, 2025))
    assert mod.load_native(str(tmp_path), {"A"}, (2016, 2025)).empty


def test_load_native_scene_keys_collapse_by_max_per_date(mod, tmp_path):
    (tmp_path / "ssebop_etf_A_no_mask_2015.csv").write_text(
        "sid,ETF_20150601,LE07_028031_20150609,LE07_028032_20150609,LC08_028031_20150617,"
        "LE07_029031_20150617\nA,0.5,0.62,0.71,0.4,\n"
    )
    nat = mod.load_native(str(tmp_path), {"A"}, (2013, 2025))
    assert nat["date"].tolist() == ["20150601", "20150609", "20150617"]
    assert nat["native"].tolist() == [0.5, 0.71, 0.4]  # max over alternates; NaN ignored
    assert nat["n_native_columns"].tolist() == [1, 2, 2]
    assert nat.loc[1, "native_columns"] == "LE07_028031_20150609;LE07_028032_20150609"
    assert nat.loc[0, "native_columns"] == "ETF_20150601"
    side = _sidecar(
        [
            ("A", "20150601", 4.0, 5.0, "s.csv"),
            ("A", "20150609", 4.0, 5.0, "s.csv"),
            ("A", "20150617", 2.0, 3.0, "s.csv"),
        ]
    )
    led = mod.convert(nat, side)
    assert led["corrected"].tolist() == pytest.approx([0.625, 0.8875, 0.6])
    assert led["n_native_columns"].tolist() == [1, 2, 2]


def test_load_native_rejects_unknown_column_and_scene_outside_year(mod, tmp_path):
    (tmp_path / "ssebop_etf_A_no_mask_2015.csv").write_text("sid,LE07_028031_20160609\nA,0.6\n")
    with pytest.raises(ValueError, match="outside file year"):
        mod.load_native(str(tmp_path), {"A"}, (2013, 2025))
    (tmp_path / "ssebop_etf_A_no_mask_2015.csv").write_text("sid,ETF_2015_06_09\nA,0.6\n")
    with pytest.raises(ValueError, match="unexpected native column"):
        mod.load_native(str(tmp_path), {"A"}, (2013, 2025))


def test_write_is_deterministic_reconstructs_and_leaves_inputs_unchanged(mod, tmp_path):
    src = tmp_path / "native" / "ssebop_etf_A_no_mask_2015.csv"
    src.parent.mkdir()
    src.write_text("sid,ETF_20150601,ETF_20150617,ETF_20151215\nA,0.5,1.0,0.4\n")
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    nat = mod.load_native(str(src.parent), {"A"}, (2013, 2025))
    side = _sidecar(
        [
            ("A", "20150601", 4.0, 5.0, "s.csv"),
            ("A", "20150617", 5.0, 6.5, "s.csv"),
            ("A", "20151215", 0.0, 1.0, "s.csv"),
        ]
    )
    led = mod.convert(nat, side)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    h1 = mod.write_grass_csvs(led, str(out1))
    h2 = mod.write_grass_csvs(led, str(out2))
    assert list(h1.values()) == list(h2.values())
    written = pd.read_csv(out1 / "ssebop_etf_grass_A_no_mask_2015.csv")
    assert written.columns.tolist() == [
        "sid",
        "ETF_20150601",
        "ETF_20150617",
    ]  # zero-ETo row omitted
    assert written.iloc[0]["ETF_20150617"] == pytest.approx(1.3)
    check = mod.reconstruct_check(led, str(out1))
    assert check["pass"] and check["n_values"] == 2
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
