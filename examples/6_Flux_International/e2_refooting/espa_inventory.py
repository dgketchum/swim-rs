"""Read-only inventory of the three ESPA order trees used by Example 6.

Every tree has the same layout (``espa_manifest.csv``, ``payloads/``, ``orders/{site}/{year}/
extract/*_ETF.tif``, ``extracts/etf_json/{site}_{year}_etf.json``). This module turns them
into tidy frames so the basis converter (Phase 2) can attach sensor and product identity to
each native SSEBop site-date, and the missing-only manifest builder (Phase 5) can classify
each PT-JPL site-date as never ordered, ordered but not delivered, or delivered but invalid.

Nothing here writes to any tree.
"""

from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

RS_ROOT = "/data/ssd1/swim/6_Flux_International/data/remote_sensing"
TREES = {
    "espa": os.path.join(RS_ROOT, "espa"),
    "espa_crop99": os.path.join(RS_ROOT, "espa_crop99"),
    "espa_ext_2008_2017": os.path.join(RS_ROOT, "espa_ext_2008_2017"),
    # Landsat 7 missing-only repair order (G6 pass 2026-09-07); scene-keyed JSON, frozen rasters
    "espa_le07_repair": os.path.join(RS_ROOT, "espa_le07_repair"),
}

PRODUCT_ID_RE = re.compile(
    r"^(LT04|LT05|LE07|LC08|LC09)_(L2SP|L2SR)_(\d{6})_(\d{8})_(\d{8})_(\d{2})_(T1|T2|RT)$"
)
ETF_TIF_RE = re.compile(r"^(?P<product_id>[A-Z0-9_]+?)_ETF\.tif$")
COLLECTION_KEYS = {
    "tm4_collection_2_l2": "LT04",
    "tm5_collection_2_l2": "LT05",
    "etm7_collection_2_l2": "LE07",
    "olitirs8_collection_2_l2": "LC08",
    "olitirs9_collection_2_l2": "LC09",
}
SENSOR_TO_COLLECTION = {v: k for k, v in COLLECTION_KEYS.items()}


def parse_product_id(product_id: str) -> dict | None:
    m = PRODUCT_ID_RE.match(product_id)
    if not m:
        return None
    sensor, level, pathrow, acquired, generated, collection, tier = m.groups()
    return {
        "product_id": product_id,
        "sensor": sensor,
        "level": level,
        "pathrow": pathrow,
        "acquired": acquired,
        "generated": generated,
        "collection": collection,
        "tier": tier,
    }


def delivered_tifs(trees: dict[str, str] = TREES) -> pd.DataFrame:
    """One row per delivered ``*_ETF.tif`` across all trees."""
    rows = []
    for tree, root in trees.items():
        for path in glob.glob(os.path.join(root, "orders", "*", "*", "extract", "*_ETF.tif")):
            parts = path.split(os.sep)
            site, year = parts[-4], parts[-3]
            m = ETF_TIF_RE.match(os.path.basename(path))
            parsed = parse_product_id(m.group("product_id")) if m else None
            if parsed is None:
                raise ValueError(f"unparseable ETF tif name: {path}")
            rows.append({"tree": tree, "site": site, "year": year, "tif_path": path, **parsed})
    cols = [
        "tree",
        "site",
        "year",
        "product_id",
        "sensor",
        "level",
        "pathrow",
        "acquired",
        "generated",
        "collection",
        "tier",
        "tif_path",
    ]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["site", "acquired", "product_id", "tree"])
        .reset_index(drop=True)
    )


def payload_requests(trees: dict[str, str] = TREES) -> pd.DataFrame:
    """One row per product ID listed in any payload JSON (the ordered-scene record)."""
    rows = []
    for tree, root in trees.items():
        for path in sorted(glob.glob(os.path.join(root, "payloads", "*_payload.json"))):
            stem = os.path.basename(path)[: -len("_payload.json")]
            site, year = stem.rsplit("_", 1)
            with open(path) as fh:
                payload = json.load(fh)
            for key, sensor in COLLECTION_KEYS.items():
                block = payload.get(key)
                if not block:
                    continue
                for pid in block.get("inputs", []):
                    parsed = parse_product_id(pid)
                    if parsed is None:
                        raise ValueError(f"unparseable product id {pid!r} in {path}")
                    if parsed["sensor"] != sensor:
                        raise ValueError(f"{pid} listed under {key} in {path}")
                    rows.append(
                        {
                            "tree": tree,
                            "site": site,
                            "year": year,
                            "payload_path": path,
                            "products": ";".join(block.get("products", [])),
                            **parsed,
                        }
                    )
    cols = [
        "tree",
        "site",
        "year",
        "product_id",
        "sensor",
        "level",
        "pathrow",
        "acquired",
        "generated",
        "collection",
        "tier",
        "products",
        "payload_path",
    ]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["site", "acquired", "product_id", "tree"])
        .reset_index(drop=True)
    )


def manifest_rows(trees: dict[str, str] = TREES) -> pd.DataFrame:
    """Site-year order bookkeeping from every ``espa_manifest.csv``."""
    frames = []
    for tree, root in trees.items():
        path = os.path.join(root, "espa_manifest.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str)
        df["tree"] = tree
        frames.append(df)
    keep = [
        "tree",
        "site",
        "year",
        "n_scenes",
        "order_id",
        "order_status",
        "download_status",
        "extract_status",
        "csv_status",
        "scene_csv",
        "payload_json",
        "extent_csv",
    ]
    out = pd.concat(frames, ignore_index=True)
    for c in keep:
        if c not in out.columns:
            out[c] = None
    return out[keep].sort_values(["site", "year", "tree"]).reset_index(drop=True)


def json_values(trees: dict[str, str] = TREES) -> pd.DataFrame:
    """Every extracted statistic across all trees (``extracts/etf_json``).

    Handles both the legacy date-keyed schema (one value per site-date, ``product_id`` None)
    and the scene-keyed schema written since 2026-09-04 (one value per product ID).
    """
    rows = []
    for tree, root in trees.items():
        for path in sorted(glob.glob(os.path.join(root, "extracts", "etf_json", "*_etf.json"))):
            stem = os.path.basename(path)[: -len("_etf.json")]
            site, year = stem.rsplit("_", 1)
            with open(path) as fh:
                data = json.load(fh)
            for key_site, entries in data.items():
                for key, stats in entries.items():
                    legacy = "date" not in stats
                    rows.append(
                        {
                            "tree": tree,
                            "site": key_site,
                            "file_site": site,
                            "year": year,
                            "date": (key if legacy else stats["date"]).replace("-", ""),
                            "product_id": None if legacy else key,
                            "mean": stats.get("mean"),
                            "count": stats.get("count"),
                            "json_path": path,
                        }
                    )
    cols = ["tree", "site", "file_site", "year", "date", "product_id", "mean", "count", "json_path"]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["site", "date", "tree"])
        .reset_index(drop=True)
    )


def scene_identity_by_site_date(tifs: pd.DataFrame) -> pd.DataFrame:
    """Collapse delivered tifs to one row per (site, acquired date).

    ``sensor`` is the single sensor when unambiguous, otherwise ``mixed``; ``product_ids`` lists
    every distinct product delivered for that site-date across trees.
    """
    if tifs.empty:
        return pd.DataFrame(
            columns=["site", "date", "sensor", "product_ids", "n_products", "trees"]
        )
    g = tifs.groupby(["site", "acquired"], sort=True)
    out = (
        pd.DataFrame(
            {
                "sensor": g["sensor"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed"),
                "product_ids": g["product_id"].agg(lambda s: ";".join(sorted(set(s)))),
                "n_products": g["product_id"].nunique(),
                "trees": g["tree"].agg(lambda s: ";".join(sorted(set(s)))),
            }
        )
        .reset_index()
        .rename(columns={"acquired": "date"})
    )
    return out
