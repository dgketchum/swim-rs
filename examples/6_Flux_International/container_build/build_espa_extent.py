# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas",
#   "pyproj",
#   "shapely",
#   "fiona",
# ]
# ///
"""Build ESPA spatial subset coordinates for a fixed-size site-centered chip.

ESPA spatial subsetting is order-level, not per-scene. The January 2026 ESPA
user guide says that if image extents are modified, an output projection must
be specified and the corner coordinates must be entered in that projection's
units. For a true 4 km x 4 km chip, the safest choice is a local UTM
projection so the chip size is exact in meters.

This utility computes a site-centered square chip in UTM meters and writes a
CSV that can be used when placing an ESPA order manually.

Example:
    uv run examples/6_Flux_International/build_espa_extent.py \
        --site US-Ro4 \
        --output /tmp/us_ro4_espa_extent.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import box

DEFAULT_SHP = Path("/data/ssd1/swim/6_Flux_International/data/gis/flux_intl_150m_23MAR2026.shp")


def _utm_crs_for_lonlat(lon: float, lat: float) -> tuple[CRS, int, str]:
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": south, "datum": "WGS84"})
    hemisphere = "south" if south else "north"
    return crs, zone, hemisphere


def build_espa_extent(
    site: str,
    output_csv: Path,
    shapefile: Path = DEFAULT_SHP,
    chip_size_m: float = 4000.0,
) -> pd.DataFrame:
    gdf = gpd.read_file(shapefile, engine="fiona").to_crs(epsg=4326)
    site_gdf = gdf[gdf["sid"] == site]
    if site_gdf.empty:
        raise ValueError(f"Site not found in shapefile: {site}")

    geom = site_gdf.iloc[0].geometry
    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y
    utm_crs, zone, hemisphere = _utm_crs_for_lonlat(lon, lat)

    site_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(utm_crs)
    centroid_utm = site_utm.iloc[0].centroid
    half = chip_size_m / 2.0
    minx = centroid_utm.x - half
    miny = centroid_utm.y - half
    maxx = centroid_utm.x + half
    maxy = centroid_utm.y + half

    chip_utm = box(minx, miny, maxx, maxy)
    chip_geo = gpd.GeoSeries([chip_utm], crs=utm_crs).to_crs(epsg=4326).iloc[0]
    lon_min, lat_min, lon_max, lat_max = chip_geo.bounds

    out = pd.DataFrame(
        [
            {
                "site": site,
                "chip_size_m": chip_size_m,
                "output_projection": "utm",
                "utm_zone": zone,
                "utm_hemisphere": hemisphere,
                "epsg": utm_crs.to_epsg(),
                "minx": round(minx, 3),
                "miny": round(miny, 3),
                "maxx": round(maxx, 3),
                "maxy": round(maxy, 3),
                "center_lon": round(lon, 8),
                "center_lat": round(lat, 8),
                "lon_min_ref": round(lon_min, 8),
                "lat_min_ref": round(lat_min, 8),
                "lon_max_ref": round(lon_max, 8),
                "lat_max_ref": round(lat_max, 8),
            }
        ]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Site: {site}")
    print(f"Shapefile: {shapefile}")
    print(f"Chip size: {chip_size_m:.1f} m")
    print(f"Output projection: UTM zone {zone} {hemisphere} (EPSG:{utm_crs.to_epsg()})")
    print("Corner coordinates for ESPA extent fields:")
    print(f"  minx={minx:.3f}")
    print(f"  miny={miny:.3f}")
    print(f"  maxx={maxx:.3f}")
    print(f"  maxy={maxy:.3f}")
    print("Geographic reference bounds:")
    print(f"  lon_min={lon_min:.8f}, lat_min={lat_min:.8f}")
    print(f"  lon_max={lon_max:.8f}, lat_max={lat_max:.8f}")
    print(f"Wrote: {output_csv}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ESPA UTM extent coordinates for a site chip"
    )
    parser.add_argument("--site", required=True, help="Flux site ID, e.g. US-Ro4")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--shapefile", default=str(DEFAULT_SHP), help="Example 6 shapefile path")
    parser.add_argument(
        "--chip-size-m", type=float, default=4000.0, help="Square chip size in meters"
    )
    args = parser.parse_args()

    build_espa_extent(
        site=args.site,
        output_csv=Path(args.output),
        shapefile=Path(args.shapefile),
        chip_size_m=args.chip_size_m,
    )


if __name__ == "__main__":
    main()
