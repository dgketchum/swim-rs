"""Attach an IrrMapper mean-irrigated-fraction attribute to the whole Ex7 cohort.

Reads the written cohort (``applied_water_fields.fgb``), computes each field's mean
IrrMapper irrigated fraction 2000-2024 (per-year ``classification.lt(1)`` averaged),
and writes it back as ``irr_mean`` on both the fgb and shp so QGIS can style every
field by it. All 110 fields are reduced through the identical server-side stack, so
irrigated fields and rainfed controls are directly comparable: the metered irrigated
cohort should read ~1 and the controls ~0.

Uses the same chunked ``reduceRegions(...).getInfo()`` path as espa_control_irrmapper.py.

    uv run python examples/7_Applied_Water/cohort_irrmapper.py
"""

import ee
import geopandas as gpd
from select_fields import PROJECT_GIS
from shapely.geometry import mapping

from swimrs.data_extraction.ee.ee_utils import is_authorized

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
YEARS = range(2000, 2025)
CHUNK = 300
FGB = PROJECT_GIS / "applied_water_fields.fgb"
SHP = PROJECT_GIS / "applied_water_fields.shp"


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

    gdf = gpd.read_file(FGB, engine="fiona").to_crs("EPSG:4326")
    means = {}
    for start in range(0, len(gdf), CHUNK):
        sub = gdf.iloc[start : start + CHUNK]
        feats = [
            ee.Feature(_ee_geom(r.geometry), {"site_id": r.site_id}) for _, r in sub.iterrows()
        ]
        fc = ee.FeatureCollection(feats)
        res = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30).getInfo()
        for f in res["features"]:
            p = f["properties"]
            vals = [p[f"irr_{y}"] for y in YEARS if p.get(f"irr_{y}") is not None]
            means[p["site_id"]] = round(sum(vals) / len(vals), 4) if vals else None
        print(f"  chunk {start}-{start + len(sub)} of {len(gdf)} done", flush=True)

    gdf["irr_mean"] = gdf.site_id.map(means)
    gdf.to_file(FGB, driver="FlatGeobuf", engine="fiona")
    gdf.to_file(SHP, engine="fiona")

    by = gdf.groupby("crop")["irr_mean"].median()
    print("\nirr_mean (2000-2024) median by crop:")
    print(by.to_string())
    print(f"\nwrote irr_mean onto {FGB}")
    print(f"wrote irr_mean onto {SHP}")


if __name__ == "__main__":
    main()
