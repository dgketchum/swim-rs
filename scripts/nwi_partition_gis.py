"""Build the per-partition NWI field shapefile carrying its GridMET GFID.

The container build consumes two GIS paths ([paths] fields_shapefile and
[paths.conus] gridmet_mapping) that in the Esmeralda pilot were the same
file: the partition's fields with an int64 GFID column plus the GridMET
cell-center LAT/LON. This script reproduces that artifact for any
partition, reusing `partition_fields` from scripts/swim_nwi.py so unit
membership is identical to the EE extraction by construction.

GFID comes from the verified south-origin GridMET formula (1386 columns,
1/24 degree):

    lat = 25.066667 + (GFID // 1386) / 24
    lon = -124.766667 + (GFID % 1386) / 24

inverted from each field's centroid, then LAT/LON are regenerated from the
formula so they are exact cell centers rather than field centroids. When the
source carries DRI's GRIDMET_ID the two are cross-checked and disagreement is
reported (and, unless --allow-mismatch, fails the run).

Usage:
    uv run python scripts/nwi_partition_gis.py \
        --shapefile data/nwi_gis/nwi_fields.shp \
        --partitions 32019a,32019b,32019c --data-root data
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

NCOLS = 1386
LAT0 = 25.066667
LON0 = -124.766667
RES = 1.0 / 24.0


def _load_partition_fields():
    """Import partition_fields from swim_nwi.py without importing `ee`."""
    path = Path(__file__).with_name("swim_nwi.py")
    spec = importlib.util.spec_from_file_location("_swim_nwi_partition", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # earthengine-api absent
        raise SystemExit(f"cannot import swim_nwi.py: {exc}") from exc
    return module.partition_fields


def gfid_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    row = np.rint((lat - LAT0) / RES).astype(np.int64)
    col = np.rint((lon - LON0) / RES).astype(np.int64)
    return row * NCOLS + col


def lonlat_from_gfid(gfid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = LAT0 + (gfid // NCOLS) * RES
    lon = LON0 + (gfid % NCOLS) * RES
    return lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapefile", required=True, help="Statewide/source fields shapefile")
    ap.add_argument(
        "--partitions", required=True, help="Comma-separated labels, e.g. 32019a,32019b"
    )
    ap.add_argument("--data-root", default="/project/handily/swim/data")
    ap.add_argument("--feature-id", default="NWI_ID")
    ap.add_argument("--partition-col", default="FIPS")
    ap.add_argument("--dri-col", default="GRIDMET_ID", help="Source GFID column to cross-check")
    ap.add_argument(
        "--allow-mismatch", action="store_true", help="Warn instead of fail on cross-check"
    )
    args = ap.parse_args()

    partition_fields = _load_partition_fields()
    labels = [p.strip() for p in args.partitions.split(",")]

    gdf = gpd.read_file(args.shapefile, engine="fiona")
    src_crs = gdf.crs
    print(f"source: {len(gdf)} fields, crs={src_crs}")

    # partition_fields returns [(label, [feature_id, ...]), ...]
    parts = dict(partition_fields(gdf, args.feature_id, args.partition_col))
    missing = [lab for lab in labels if lab not in parts]
    if missing:
        raise SystemExit(f"partitions not produced by partition_fields: {missing}")

    indexed = gdf.set_index(args.feature_id, drop=False)
    for label in labels:
        # Reindex on the partition's id list so field order matches the EE export.
        sub = indexed.loc[parts[label]].copy().reset_index(drop=True)

        # Centroid in the projected CRS (planar and correct), then to lon/lat
        # for the GFID lookup.
        cent = sub.geometry.centroid.to_crs("EPSG:4326")
        gfid = gfid_from_lonlat(cent.x.to_numpy(), cent.y.to_numpy())

        if args.dri_col in sub.columns:
            dri = sub[args.dri_col].astype(str).str.strip()
            ok = dri.str.lstrip("-").str.isdigit()
            agree = int((gfid[ok.to_numpy()] == dri[ok].astype(np.int64).to_numpy()).sum())
            total = int(ok.sum())
            pct = 100.0 * agree / total if total else float("nan")
            print(f"{label}: GFID cross-check {agree}/{total} agree ({pct:.2f}%)")
            if total and agree != total and not args.allow_mismatch:
                bad = sub.loc[
                    ok.to_numpy() & (gfid != dri[ok].astype(np.int64).to_numpy()), args.feature_id
                ]
                raise SystemExit(
                    f"{label}: GFID mismatch for {list(bad[:10])} (use --allow-mismatch to override)"
                )

        lon, lat = lonlat_from_gfid(gfid)
        out = gpd.GeoDataFrame(
            {
                args.feature_id: sub[args.feature_id].astype(str).to_numpy(),
                "GFID": gfid,
                "STATION_ID": gfid,
                "LAT": lat,
                "LON": lon,
            },
            geometry=sub.geometry.to_numpy(),
            crs=src_crs,
        )

        gis_dir = Path(args.data_root) / label / "gis"
        gis_dir.mkdir(parents=True, exist_ok=True)
        # The TOML pipeline reads the .shp; the .fgb is the inspection copy.
        dest = gis_dir / f"nwi_fields_{label}_gfid.shp"
        out.to_file(dest, engine="fiona")
        out.to_file(dest.with_suffix(".fgb"), driver="FlatGeobuf", engine="fiona")
        print(
            f"{label}: wrote {dest} (+.fgb) — {len(out)} fields, {out.GFID.nunique()} GridMET cells"
        )


if __name__ == "__main__":
    main()
