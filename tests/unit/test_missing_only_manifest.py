"""Unit tests for the E2 missing-only ESPA manifest builder (Phase 5, Gate G5).

Covers the pure pieces of
``examples/6_Flux_International/e2_refooting/phase5_build_missing_only_manifest.py``: tier-ranked
product-ID selection, the delivery/order classification, the request rule (never-ordered,
cancelled-only, and ordered-not-delivered scenes for both sensors), and payload construction
(ETM+ ``et`` retained, extents copied verbatim).
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2_DIR = REPO_ROOT / "examples" / "6_Flux_International" / "e2_refooting"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location(
        "phase5_build_missing_only_manifest", E2_DIR / "phase5_build_missing_only_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(E2_DIR)]
    return module


def _meta(pids):
    s = pd.Series(pids)
    df = pd.DataFrame(
        {
            "product_id": s,
            "sensor": s.str[:4],
            "level": s.str[5:9],
            "pathrow": s.str[10:16],
            "acquired": s.str[17:25],
            "generated": s.str[26:34],
            "tier": s.str[-2:],
        }
    )
    df["level_rank"] = df["level"].map({"L2SP": 0, "L2SR": 1})
    df["tier_rank"] = df["tier"].map({"T1": 0, "T2": 1, "RT": 2})
    return df


def test_best_product_id_prefers_l2sp_t1_newest(mod):
    meta = _meta(
        [
            "LE07_L2SP_044033_20150601_20200903_02_T1",
            "LE07_L2SP_044033_20150601_20200801_02_T1",
            "LE07_L2SP_044033_20150601_20200903_02_T2",
            "LE07_L2SR_044033_20150601_20200903_02_T1",
            "LC08_L2SP_044033_20150609_20200903_02_T2",
        ]
    )
    scenes = pd.DataFrame(
        {
            "sensor": ["LE07", "LC08", "LC08"],
            "pathrow": ["044033", "044033", "044034"],
            "date": ["20150601", "20150609", "20150609"],
        }
    )
    best = mod.best_product_ids(scenes, meta).set_index(["sensor", "pathrow", "date"])
    assert (
        best.loc[("LE07", "044033", "20150601"), "product_id"]
        == "LE07_L2SP_044033_20150601_20200903_02_T1"
    )
    assert best.loc[("LE07", "044033", "20150601"), "n_candidates"] == 4
    assert best.loc[("LC08", "044033", "20150609"), "tier"] == "T2"
    assert pd.isna(best.loc[("LC08", "044034", "20150609"), "product_id"])
    assert best.loc[("LC08", "044034", "20150609"), "n_candidates"] == 0


def _classify_fixture(mod):
    cand = pd.DataFrame(
        {
            "site": ["S"] * 6,
            "scene_key": [
                "LE07_044033_20150601",
                "LE07_044033_20150617",
                "LC08_044033_20150609",
                "LC08_044033_20150625",
                "LE07_044033_20150703",
                "LC08_044033_20150711",
            ],
            "sensor": ["LE07", "LE07", "LC08", "LC08", "LE07", "LC08"],
            "pathrow": ["044033"] * 6,
            "date": ["20150601", "20150617", "20150609", "20150625", "20150703", "20150711"],
        }
    )
    products = pd.DataFrame(
        {
            "sensor": ["LE07", "LE07", "LC08", "LC08", "LE07", "LC08"],
            "pathrow": ["044033"] * 6,
            "date": cand["date"],
            "level": ["L2SP"] * 6,
            "tier": ["T1"] * 6,
            "n_candidates": [1, 1, 1, 1, 1, 0],
            "product_id": [
                f"{s}_L2SP_044033_{d}_20200903_02_T1"
                for s, d in zip(cand["sensor"], cand["date"], strict=True)
            ],
        }
    )
    products.loc[5, "product_id"] = None
    tifs = pd.DataFrame(
        {
            "site": ["S", "S"],
            "sensor": ["LE07", "LC08"],
            "pathrow": ["044033", "044033"],
            "acquired": ["20150617", "20150625"],
            "tree": ["espa", "espa"],
            "tif_path": ["a.tif", "b.tif"],
        }
    )
    payloads = pd.DataFrame(
        {
            "tree": ["espa", "espa", "espa_crop99"],
            "site": ["S", "S", "S"],
            "year": ["2015", "2015", "2015"],
            "sensor": ["LC08", "LE07", "LE07"],
            "pathrow": ["044033"] * 3,
            "acquired": ["20150609", "20150703", "20150703"],
        }
    )
    manifests = pd.DataFrame(
        {
            "tree": ["espa", "espa_crop99"],
            "site": ["S", "S"],
            "year": ["2015", "2015"],
            "order_id": ["espa-1", "espa-2"],
            "order_status": ["ready_for_download", "cancelled"],
        }
    )
    jvals = pd.DataFrame({"tree": ["espa"], "site": ["S"], "date": ["20150625"], "mean": [0.7]})
    return mod.classify(cand, tifs, payloads, manifests, jvals, products).set_index("scene_key")


def test_classification_statuses(mod):
    c = _classify_fixture(mod)
    assert c.loc["LE07_044033_20150601", "status"] == "never_ordered"
    assert c.loc["LE07_044033_20150617", "status"] == "delivered_nodata"
    assert c.loc["LC08_044033_20150609", "status"] == "ordered_not_delivered"
    assert c.loc["LC08_044033_20150625", "status"] == "delivered_not_ingested"
    assert (
        c.loc["LE07_044033_20150703", "status"] == "ordered_not_delivered"
    )  # one live order outweighs a cancelled one
    assert c.loc["LC08_044033_20150711", "status"] == "no_product_id"


def test_request_rule_le07_and_oli(mod):
    c = _classify_fixture(mod)
    assert c.loc["LE07_044033_20150601", "request"]
    assert c.loc["LE07_044033_20150703", "request"]  # LE07 ordered-not-delivered is requested
    # OLI ordered-not-delivered is requested too (user decision 2026-09-04); status label kept
    assert c.loc["LC08_044033_20150609", "request"]
    assert c.loc["LC08_044033_20150609", "status"] == "ordered_not_delivered"
    assert not c.loc["LE07_044033_20150617", "request"]  # delivered_nodata stays residual
    assert not c.loc["LC08_044033_20150625", "request"]  # delivered_not_ingested stays residual
    assert not c.loc["LC08_044033_20150711", "request"]  # no product id
    assert c.loc["LE07_044033_20150601", "selection_reason"] == "ptjpl_only_le07_never_ordered"
    assert (
        c.loc["LC08_044033_20150609", "selection_reason"] == "ptjpl_only_lc08_ordered_not_delivered"
    )
    assert c.loc["LE07_044033_20150617", "selection_reason"] == "residual_le07_delivered_nodata"


def test_cancelled_only_orders_are_requestable(mod):
    cand = pd.DataFrame(
        {
            "site": ["S"],
            "scene_key": ["LC08_044033_20150609"],
            "sensor": ["LC08"],
            "pathrow": ["044033"],
            "date": ["20150609"],
        }
    )
    products = pd.DataFrame(
        {
            "sensor": ["LC08"],
            "pathrow": ["044033"],
            "date": ["20150609"],
            "level": ["L2SP"],
            "tier": ["T1"],
            "n_candidates": [1],
            "product_id": ["LC08_L2SP_044033_20150609_20200903_02_T1"],
        }
    )
    payloads = pd.DataFrame(
        {
            "tree": ["espa"],
            "site": ["S"],
            "year": ["2015"],
            "sensor": ["LC08"],
            "pathrow": ["044033"],
            "acquired": ["20150609"],
        }
    )
    manifests = pd.DataFrame(
        {
            "tree": ["espa"],
            "site": ["S"],
            "year": ["2015"],
            "order_id": ["espa-1"],
            "order_status": ["cancelled"],
        }
    )
    tifs = pd.DataFrame(columns=["site", "sensor", "pathrow", "acquired", "tree", "tif_path"])
    jvals = pd.DataFrame(columns=["tree", "site", "date", "mean"])
    c = mod.classify(cand, tifs, payloads, manifests, jvals, products).iloc[0]
    assert c["status"] == "ordered_cancelled" and c["request"]


def test_payload_keeps_etm7_et_and_extent(mod):
    extent = {
        "utm_zone": 10,
        "utm_hemisphere": "north",
        "minx": 1.0,
        "miny": 2.0,
        "maxx": 3.0,
        "maxy": 4.0,
    }
    pids = [
        "LE07_L2SP_044033_20150601_20200903_02_T1",
        "LC08_L2SP_044033_20150609_20200903_02_T1",
        "LE07_L2SP_044033_20150601_20200903_02_T1",
    ]
    p = mod.build_payload("S", 2015, pids, extent)
    assert p["etm7_collection_2_l2"] == {
        "inputs": ["LE07_L2SP_044033_20150601_20200903_02_T1"],
        "products": ["et"],
    }
    assert p["olitirs8_collection_2_l2"]["products"] == ["et"]
    assert p["image_extents"] == {
        "north": 4.0,
        "south": 2.0,
        "east": 3.0,
        "west": 1.0,
        "units": "meters",
    }
    assert p["projection"] == {"utm": {"zone": 10, "zone_ns": "north"}}


def test_sample_for_availability_spans_sensors_and_dates(mod):
    req = pd.DataFrame(
        {
            "sensor": ["LE07"] * 10 + ["LC08"] * 4,
            "date": [f"20{13 + i}0101" for i in range(10)]
            + ["20150101", "20160101", "20170101", "20180101"],
            "product_id": [f"P{i}" for i in range(14)],
        }
    )
    s = mod.sample_for_availability(req, 6)
    assert "P0" in s and "P9" in s  # earliest and latest LE07
    assert any(p in s for p in ["P10", "P11", "P12", "P13"])
