"""Build the IrrMapper rainfed cache for the ESPA control candidate pool.

For every field-sized, compact ESPA 2015 non-irrigated polygon (the pool defined by
``select_fields.espa_control_candidates``), computes the per-year IrrMapper irrigated
fraction 2000-2024 and writes one row per field keyed by ``fid2015``:

    fid2015, f_acres, mean_irr, max_irr

``select_fields.select_espa_controls`` joins on this and keeps only strictly
never-irrigated fields (``max_irr == 0``). Run this whenever the candidate pool
definition (ACRE_MIN, CONTROL_ACRE_MAX, PP_MIN) or the source shapefile changes,
before regenerating the cohort.

Uses the same server-side per-year ``classification.lt(1)`` stack + chunked
``reduceRegions(...).getInfo()`` path as the properties/IrrMapper extraction.
CO and ID are both inside IrrMapper's western-11 coverage.

    uv run python examples/7_Applied_Water/espa_control_irrmapper.py
"""

import ee
import pandas as pd
from select_fields import ESPA_CONTROL_IRR, espa_control_candidates
from shapely.geometry import mapping

from swimrs.data_extraction.ee.ee_utils import is_authorized

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
YEARS = range(2000, 2025)
CHUNK = 300


def _ee_geom(geom):
    g = mapping(geom)
    if g["type"] == "Polygon":
        return ee.Geometry.Polygon(g["coordinates"])
    return ee.Geometry.MultiPolygon(g["coordinates"])


def main() -> None:
    is_authorized()
    coll = ee.ImageCollection(IRR)
    img = None
    for y in YEARS:
        a = coll.filterDate(f"{y}-01-01", f"{y}-12-31").select("classification").mosaic()
        b = a.lt(1).rename(f"irr_{y}").float()  # class 0 = irrigated
        img = b if img is None else img.addBands(b)

    cand = espa_control_candidates().to_crs("EPSG:4326")
    print(f"candidate pool: {len(cand)} field-sized compact non-irrigated polygons")

    rows = []
    fids = cand.fid2015.tolist()
    for start in range(0, len(cand), CHUNK):
        sub = cand.iloc[start : start + CHUNK]
        feats = [
            ee.Feature(_ee_geom(r.geometry), {"fid2015": int(r.fid2015)}) for _, r in sub.iterrows()
        ]
        fc = ee.FeatureCollection(feats)
        res = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30).getInfo()
        for f in res["features"]:
            rows.append(f["properties"])
        print(f"  chunk {start}-{start + len(sub)} of {len(fids)} done", flush=True)

    df = pd.DataFrame(rows)
    irr_cols = [f"irr_{y}" for y in YEARS if f"irr_{y}" in df.columns]
    df["mean_irr"] = df[irr_cols].mean(axis=1)
    df["max_irr"] = df[irr_cols].max(axis=1)
    df = df.merge(cand[["fid2015", "f_acres"]], on="fid2015", how="left")
    cols = ["fid2015", "f_acres", "mean_irr", "max_irr"] + irr_cols
    df[cols].to_csv(ESPA_CONTROL_IRR, index=False)

    strict = int((df.max_irr == 0.0).sum())
    print(f"\nstrictly never-irrigated (max_irr==0, 2000-2024): {strict}/{len(df)}")
    print(f"wrote {ESPA_CONTROL_IRR}")


if __name__ == "__main__":
    main()
