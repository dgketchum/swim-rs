"""Write the Example 10 field polygons and single-cell GridMET mapping.

Reads the tracked FlatGeobuf (24 OWRD fields near Pendleton, OR), writes the
fields shapefile the TOML points at (plus a .fgb beside it for inspection), and
writes the GridMET mapping shapefile with every field assigned to cell 680670.
The mapping carries LAT/LON of the cell center and ELEV from the GridMET
elevation grid, which is what ``download_gridmet`` reads.

    uv run python examples/10_OR_Irrigation_Test/prepare_fields.py [--source FGB]
"""

import argparse
import os

import geopandas as gpd
from ex10_paths import GRIDMET_CELL, GRIDMET_CELL_LAT, GRIDMET_CELL_LON, HERE, load_config

DEFAULT_SOURCE = HERE / "data" / "gis" / "or_irr_fields.fgb"
KEEP_COLS = [
    "OPENET_ID",
    "fid",
    "Acres",
    "HUC8",
    "HUC12",
    "GRIDMET_ID",
    "OWRD",
    "ITYPE",
    "srctype",
    "IRR_EFF",
]


def write_fields(source: str, fields_shp: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(source, engine="fiona")
    missing = [c for c in KEEP_COLS if c not in gdf.columns]
    if missing:
        raise ValueError(f"{source} lacks expected columns: {missing}")
    gdf = gdf[KEEP_COLS + ["geometry"]].copy()
    gdf["OPENET_ID"] = gdf["OPENET_ID"].astype(str)
    if gdf["OPENET_ID"].duplicated().any():
        raise ValueError("OPENET_ID is not unique")
    gdf["state"] = "OR"
    gdf = gdf.to_crs("EPSG:5071")

    os.makedirs(os.path.dirname(fields_shp), exist_ok=True)
    gdf.to_file(fields_shp, engine="fiona")
    gdf.to_file(fields_shp.replace(".shp", ".fgb"), driver="FlatGeobuf", engine="fiona")
    print(f"Wrote {len(gdf)} fields -> {fields_shp} (+ .fgb)")
    return gdf


def write_gridmet_mapping(gdf: gpd.GeoDataFrame, mapping_shp: str) -> None:
    from swimrs.data_extraction.gridmet.thredds import GridMet

    elev = GridMet("elev", lat=GRIDMET_CELL_LAT, lon=GRIDMET_CELL_LON).get_point_elevation()
    out = gdf.copy()
    out["GFID"] = GRIDMET_CELL
    out["STATION_ID"] = GRIDMET_CELL
    out["LAT"] = GRIDMET_CELL_LAT
    out["LON"] = GRIDMET_CELL_LON
    out["ELEV"] = float(elev)
    out.to_file(mapping_shp, engine="fiona")
    print(
        f"Wrote GridMET mapping -> {mapping_shp}: all {len(out)} fields on cell "
        f"{GRIDMET_CELL} ({GRIDMET_CELL_LAT}, {GRIDMET_CELL_LON}), elev {elev:.1f} m"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input FlatGeobuf")
    args = p.parse_args()

    cfg = load_config()
    gdf = write_fields(args.source, cfg.fields_shapefile)
    write_gridmet_mapping(gdf, cfg.gridmet_mapping_shp)


if __name__ == "__main__":
    main()
