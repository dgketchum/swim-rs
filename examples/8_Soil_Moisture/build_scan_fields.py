"""Build the Example 8 SCAN soil-moisture validation cohort.

Turns the 28 screened USDA-NRCS SCAN cropland stations (ISMN archive, FROM-GLC10
cropland gate; see notes/soil_moisture_networks/scan_ACQUISITION.md) into a SWIM
fields shapefile. Each station is a point coordinate, so the "field" SWIM extracts
over is a circular buffer around the station (BUFFER_M, default 150 m radius ->
~7 ha / ~10x10 Landsat-30m pixels). This is the field-scale footprint the getInfo
extraction averages Landsat/Sentinel NDVI and OpenET ETf over, mirroring the
~50 m flux footprints of Example 5 but larger, per project direction.

In-situ soil moisture (theta) is validation-only and is NEVER a model input; the
observed cleaned theta series live at /data/ssd1/swim/soil_moisture/scan/ and are
joined only at evaluation (evaluate.py) against modeled theta_avail.

SCAN UIDs (`SCAN:Lind#1`) contain characters unsafe for filenames/DBF, so the
feature id `site_id` is a sanitized token (`Lind1`); the original UID is kept in
`scan_uid` and in the sidecar scan_sites.csv so observed theta can be re-joined.

Outputs (project gis dir + a versioned copy in this example's data dir):
    scan_fields.shp / .fgb   feature id `site_id`; also scan_uid, state, lat, lon,
                             irrig, suit, glc10, geometry (EPSG:5071)
    scan_sites.csv           site_id, scan_uid, lat, lon, state, irrig_status,
                             suitability, theta_csv  (validation join table)
    notes/selection_qc.md    cohort counts + tier / irrigation / period summary

    uv run python examples/8_Soil_Moisture/build_scan_fields.py [--buffer-m 150]
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path("/home/dgketchum/code/swim-rs")
EXAMPLE_DIR = REPO / "examples" / "8_Soil_Moisture"
SHORTLIST = EXAMPLE_DIR / "notes" / "soil_moisture_networks" / "scan_shortlist.csv"
THETA_DIR = Path("/data/ssd1/swim/soil_moisture/scan")
PROJECT_GIS = Path("/data/ssd1/swim/8_Soil_Moisture/data/gis")

EQUAL_AREA = (
    "EPSG:5071"  # NAD83 CONUS Albers — accurate metric buffering; matches Example 5 flux_fields.shp
)
BUFFER_M = 150.0  # buffer radius in meters around each SCAN station point

# SCAN station -> USPS state (from station coordinates). Cosmetic only: the
# GridMET mapping keys on lat/lon and the getInfo extraction ignores state_col
# (state_col feeds only the unused bucket-NDVI path).
STATE = {
    "Circleville": "UT",
    "Lind1": "WA",
    "TableMountain": "MT",
    "Morgan": "UT",
    "Manderfield": "UT",
    "BraggFarm": "AL",
    "CrescentLake1": "MN",
    "UapbEarle": "AR",
    "PeeDee": "SC",
    "SandyRidge": "MS",
    "Dexter": "MO",
    "Ames": "IA",
    "Vance": "MS",
    "Tuskegee": "AL",
    "Spickard": "MO",
    "UapbDewitt": "AR",
    "Princeton1": "KY",
    "MorrisFarms": "AL",
    "PerdidoRivFarms": "AL",
    "Mayday": "MS",
    "MollyCaren1": "OH",
    "MarkTwainHS": "MO",
    "Vallecitos": "CA",
    "CookFarmFieldD": "WA",
    "JohnsonFarm": "NE",
    "UapbLonokeFarm": "AR",
    "ShagbarkHills": "IA",
    "EvergladesARS": "FL",
}


def _sanitize(uid: str) -> str:
    """`SCAN:Lind#1` -> `Lind1` (drop network prefix, keep alnum only)."""
    name = uid.split(":", 1)[1] if ":" in uid else uid
    return re.sub(r"[^0-9A-Za-z]", "", name)


def build(buffer_m: float = BUFFER_M) -> gpd.GeoDataFrame:
    sl = pd.read_csv(SHORTLIST)
    sl["site_id"] = sl["station_ui"].map(_sanitize)
    if sl["site_id"].duplicated().any():
        dups = sl.loc[sl["site_id"].duplicated(keep=False), "station_ui"].tolist()
        raise SystemExit(f"Sanitized site_id collision: {dups}")

    pts = gpd.GeoDataFrame(
        sl.copy(),
        geometry=gpd.points_from_xy(sl["longitude"], sl["latitude"]),
        crs="EPSG:4326",
    ).to_crs(EQUAL_AREA)
    pts["geometry"] = pts.geometry.buffer(buffer_m)

    theta_csv = {s: (THETA_DIR / f"SCAN_{s.split(':', 1)[1]}.parquet") for s in sl["station_ui"]}
    fields = gpd.GeoDataFrame(
        {
            "site_id": sl["site_id"].values,
            "scan_uid": sl["station_ui"].values,
            "lc_class": "Croplands",
            "state": [STATE.get(s, "XX") for s in sl["site_id"]],
            "source": "SCAN",
            "record": [f"{a}-{b}" for a, b in zip(sl["period_start"], sl["period_end"])],
            # station coords kept under non-colliding names: assign_gridmet_ids adds
            # uppercase LAT/LON/ELEV, and the DBF driver is case-insensitive, so
            # lowercase lat/lon/elev would collide -> LAT_1 and break download_gridmet.
            "site_lat": sl["latitude"].values,
            "site_lon": sl["longitude"].values,
            "irrig": sl["irrig_status"].values,
            "suit": sl["suitability"].values,
            "glc10": sl["glc10_label"].values,
            "geometry": pts.geometry.values,
        },
        crs=EQUAL_AREA,
    )
    missing = [s for s in fields.site_id if s not in STATE]
    if missing:
        print(f"  WARNING: no state mapping for {missing} (cosmetic, set 'XX')")

    sites = pd.DataFrame(
        {
            "site_id": fields.site_id,
            "scan_uid": fields.scan_uid,
            "lat": fields.site_lat,
            "lon": fields.site_lon,
            "state": fields.state,
            "irrig_status": fields.irrig,
            "suitability": fields.suit,
            "theta_csv": [str(theta_csv[u]) for u in fields.scan_uid],
        }
    )
    return fields, sites


def _write_qc(fields: gpd.GeoDataFrame, path: Path, buffer_m: float) -> None:
    lines = [
        "# Example 8 SCAN cohort selection QC\n",
        f"Cohort: {len(fields)} SCAN cropland stations "
        f"(150 m -> {buffer_m:.0f} m buffer footprints, "
        f"{3.14159 * (buffer_m**2) / 1e4:.1f} ha each)\n",
        f"By state: {fields.state.value_counts().to_dict()}\n",
        f"By irrigation label: {fields.irrig.value_counts().to_dict()}\n",
        f"Suitability — median {fields.suit.median():.1f}, "
        f"range {fields.suit.min():.1f}-{fields.suit.max():.1f}\n",
        "\nIn-situ soil moisture is validation-only (evaluate.py), never a model input.\n",
        "Irrigation labels are provisional climate proxies (see scan_ACQUISITION.md) "
        "and are NOT used to configure the model — irrigation status comes from the "
        "internal water-balance algorithm.\n",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer-m", type=float, default=BUFFER_M, help="buffer radius (m)")
    args = ap.parse_args()

    fields, sites = build(args.buffer_m)

    PROJECT_GIS.mkdir(parents=True, exist_ok=True)
    (EXAMPLE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (EXAMPLE_DIR / "notes").mkdir(parents=True, exist_ok=True)

    shp = PROJECT_GIS / "scan_fields.shp"
    fgb = PROJECT_GIS / "scan_fields.fgb"
    fields.to_file(shp, engine="fiona")
    fields.to_file(fgb, driver="FlatGeobuf", engine="fiona")
    sites.to_csv(EXAMPLE_DIR / "data" / "scan_sites.csv", index=False)
    _write_qc(fields, EXAMPLE_DIR / "notes" / "selection_qc.md", args.buffer_m)

    print(f"wrote {shp}  ({len(fields)} fields, EPSG:5071)")
    print(f"wrote {fgb}")
    print(f"wrote {EXAMPLE_DIR / 'data' / 'scan_sites.csv'}")
    print(f"wrote {EXAMPLE_DIR / 'notes' / 'selection_qc.md'}")


if __name__ == "__main__":
    main()
