"""Pull WaterRightPous *polygon geometry* for every RightID referenced by WMIS.

Companion to ``download_pou.py`` (which fetches attributes only, with
``returnGeometry:false``). The Example 7 applied-water validation needs the
place-of-use polygon geometry so each metered POD can be tied to a real field
boundary. Output: ``pou_polygons.fgb`` (EPSG:4326), one feature per POU polygon,
keyed on ``RightID``.

    uv run python data/idwr_wmis/download_pou_geom.py

Read-only GET against the public IDWR ArcGIS FeatureServer.
"""

import json
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
URL = "https://gis.idwr.idaho.gov/hosting/rest/services/Allocation/WaterRightPous/FeatureServer/0/query"
H = {"User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"}
OUT_FIELDS = "RightID,WaterRightNumber,BasinNumber,WaterUse,WaterUseCode,TotalAcres,AcreLimit,Owner,Status,Source"


def _right_ids() -> list[int]:
    d = json.load(open(HERE / "wmis_raw.json"))
    rids = set()
    for f in d["features"]:
        r = f["attributes"].get("RightIDs")
        if r:
            for tok in re.split(r"[,\s]+", str(r).strip()):
                if tok.isdigit():
                    rids.add(int(tok))
    return sorted(rids)


def _fetch_chunk(sess: requests.Session, where: str) -> list[dict]:
    """Fetch all GeoJSON features for a where-clause, paging past transfer limits."""
    feats: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": OUT_FIELDS,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
        }
        for attempt in range(4):
            try:
                rr = sess.get(URL, params=params, headers=H, timeout=180)
                j = rr.json()
                if "features" in j:
                    break
                raise ValueError(j.get("error", j))
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        batch = j.get("features", [])
        feats.extend(batch)
        if not j.get("properties", {}).get("exceededTransferLimit") and not j.get(
            "exceededTransferLimit"
        ):
            break
        if not batch:
            break
        offset += len(batch)
    return feats


def main() -> None:
    rids = _right_ids()
    print("unique RightIDs to query:", len(rids))

    sess = requests.Session()
    all_feats: list[dict] = []
    CH = 200
    n_batches = (len(rids) + CH - 1) // CH
    for i in range(0, len(rids), CH):
        chunk = rids[i : i + CH]
        where = "RightID IN (" + ",".join(str(x) for x in chunk) + ")"
        feats = _fetch_chunk(sess, where)
        all_feats.extend(feats)
        print(f"  batch {i // CH + 1}/{n_batches}: +{len(feats)} (cum {len(all_feats)})")

    if not all_feats:
        raise SystemExit("No POU polygons returned — check the service / RightIDs.")

    gdf = gpd.GeoDataFrame.from_features(all_feats, crs="EPSG:4326")
    # Drop null/empty geometries defensively (some POUs are unmapped).
    n0 = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    print(f"\nPOU polygons: {len(gdf)} (dropped {n0 - len(gdf)} null/empty)")
    print("distinct RightID:", gdf["RightID"].nunique() if "RightID" in gdf else "n/a")
    if "TotalAcres" in gdf:
        ta = pd.to_numeric(gdf["TotalAcres"], errors="coerce")
        print("TotalAcres median:", round(float(ta.median()), 1))

    out = HERE / "pou_polygons.fgb"
    gdf.to_file(out, driver="FlatGeobuf")
    print("wrote", out)


if __name__ == "__main__":
    main()
