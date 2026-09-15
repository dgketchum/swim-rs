# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "geopandas",
#   "rasterio",
#   "rasterstats",
#   "shapely",
#   "fiona",
# ]
# ///
"""Extract ETF values from downloaded ESPA rasters to site polygons.

Scans extracted order directories for *_ETF.tif files, parses the Landsat
product ID from each filename, runs zonal_stats on the site polygon, and
writes per-site-year intermediate JSON files keyed by **product ID**, so a
site imaged twice on one date (adjacent rows of the same overpass) keeps both
values under their own scene identity:

    {site: {product_id: {date, sensor, pathrow, count, mean, std, min, max, tif}}}

The date-keyed schema written before 2026-09-04 (``{site: {YYYY-MM-DD: stats}}``)
kept only the last-scanned scene of a same-date pair. When an existing JSON in
that legacy schema is re-extracted it is replaced wholesale by the scene-keyed
result (the whole extract directory is rescanned every run), and the
replacement is reported.

Example:
    uv run examples/6_Flux_International/espa/espa_extract_etf.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import mapping

DEFAULT_SHP = Path("/data/ssd1/swim/6_Flux_International/data/gis/flux_intl_150m_23MAR2026.shp")

# Full Collection 2 Level-2 product ID, e.g. LC08_L2SP_042028_20200715_20200722_02_T1_ETF.tif
ETF_RE = re.compile(
    r"^(?P<product_id>(?P<sensor>LT04|LT05|LE07|LC08|LC09)_L2\w{2}_(?P<pathrow>\d{6})_"
    r"(?P<date>\d{8})_\d{8}_\d{2}_\w+?)_ETF\.tif$"
)
LEGACY_DATE_KEY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ETF_SCALE_FACTOR = 0.0001
PLAUSIBLE_ETF_RANGE = (-0.1, 2.5)


def _load_site_geometries(shapefile: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shapefile, engine="fiona").to_crs(epsg=4326)
    return gdf.set_index("sid", drop=False)


def _find_etf_tifs(extract_dir: Path) -> list[tuple[Path, dict]]:
    """Every ``*_ETF.tif`` with its parsed identity, ordered by (date, product_id)."""
    results = []
    if not extract_dir.exists():
        return results
    for tif in extract_dir.rglob("*_ETF.tif"):
        m = ETF_RE.match(tif.name)
        if m:
            results.append((tif, m.groupdict()))
    return sorted(results, key=lambda x: (x[1]["date"], x[1]["product_id"]))


def is_legacy_date_keyed(values: dict) -> bool:
    return bool(values) and all(LEGACY_DATE_KEY_RE.match(k) for k in values)


def _reproject_polygon(geom_4326, raster_crs) -> dict:
    """Reproject a 4326 geometry to the raster's CRS and return GeoJSON mapping."""
    gs = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(raster_crs)
    return mapping(gs.iloc[0])


def extract_site_year(
    site: str,
    extract_dir: Path,
    site_geom_4326,
) -> dict[str, dict]:
    tifs = _find_etf_tifs(extract_dir)
    if not tifs:
        return {}

    # Cache reprojected polygon per raster CRS
    reprojected_cache: dict[str, dict] = {}
    scene_values: dict[str, dict] = {}

    for tif_path, ident in tifs:
        with rasterio.open(tif_path) as src:
            raster_crs = src.crs

        crs_key = str(raster_crs)
        if crs_key not in reprojected_cache:
            reprojected_cache[crs_key] = _reproject_polygon(site_geom_4326, raster_crs)
        poly_native = reprojected_cache[crs_key]

        stats = zonal_stats(
            poly_native,
            str(tif_path),
            stats=["count", "mean", "std", "min", "max"],
            geojson_out=False,
        )
        if not stats or not stats[0] or stats[0].get("count", 0) == 0:
            continue

        result = stats[0]
        # Apply ESPA scale factor to convert raw integers to physical ETF
        for key in ("mean", "std", "min", "max"):
            if result.get(key) is not None:
                result[key] = result[key] * ETF_SCALE_FACTOR

        mean_val = result.get("mean")
        if mean_val is not None:
            if mean_val < PLAUSIBLE_ETF_RANGE[0] or mean_val > PLAUSIBLE_ETF_RANGE[1]:
                continue

        d = ident["date"]
        if ident["product_id"] in scene_values:
            raise ValueError(f"{site}: product {ident['product_id']} extracted twice ({tif_path})")
        scene_values[ident["product_id"]] = {
            "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
            "sensor": ident["sensor"],
            "pathrow": ident["pathrow"],
            "tif": tif_path.name,
            **result,
        }

    return scene_values


def extract_all(manifest_path: Path, shapefile: Path = DEFAULT_SHP) -> None:
    manifest = pd.read_csv(manifest_path, dtype=str)
    site_gdf = _load_site_geometries(shapefile)
    extracts_dir = manifest_path.parent / "extracts" / "etf_json"
    extracts_dir.mkdir(parents=True, exist_ok=True)

    # Process any fully downloaded site-year, including re-extraction
    downloadable = manifest[manifest["download_status"] == "complete"]

    if downloadable.empty:
        print("No downloaded orders to process.")
        return

    missing = sorted(set(downloadable["site"]) - set(site_gdf.index))
    if missing:
        # A downloaded site with no geometry means the wrong shapefile was passed. Refuse
        # to run rather than silently leaving those site-years unextracted (the pre-2026-09-07
        # behaviour, which left 240 legacy site-years with no ETF record).
        raise ValueError(
            f"{len(missing)} downloaded site(s) absent from {shapefile}: {', '.join(missing)}"
        )

    print(f"Extracting ETF from {len(downloadable)} site-years...")

    for idx, row in downloadable.iterrows():
        site = row["site"]
        year = row["year"]
        extract_dir = Path(row["output_dir"]) / "extract"

        json_path = extracts_dir / f"{site}_{year}_etf.json"

        # Load existing extractions to merge incrementally
        existing: dict[str, dict] = {}
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            existing = data.get(site, {})

        site_geom = site_gdf.loc[site].geometry
        values = extract_site_year(site, extract_dir, site_geom)

        if is_legacy_date_keyed(existing):
            # Pre-2026-09-04 schema kept one value per date; the rescan above holds every
            # scene, so the legacy dict is superseded rather than merged.
            print(f"  {site}/{year}: legacy date-keyed json ({len(existing)} dates) superseded")
            existing = {}

        # Merge by product ID: a re-extracted scene replaces its own earlier value only
        merged = {**existing, **values}
        changed = merged != existing

        if merged:
            n_new = len(set(merged) - set(existing))
            if changed or not json_path.exists():
                with open(json_path, "w") as f:
                    json.dump({site: merged}, f, indent=2)
                n_dates = len({v["date"] for v in merged.values()})
                if n_new > 0 or not existing:
                    print(f"  {site}/{year}: {len(merged)} scenes / {n_dates} dates ({n_new} new)")
                else:
                    print(f"  {site}/{year}: updated {len(merged)} scenes (value corrections only)")
                # Reset csv_status so CSV writer regenerates from updated JSON
                if "csv_status" in manifest.columns:
                    manifest.at[idx, "csv_status"] = ""
            manifest.at[idx, "extract_status"] = "etf_extracted"
        else:
            print(f"  {site}/{year}: no valid ETF observations")
            manifest.at[idx, "extract_status"] = "no_etf"

    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest updated: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ETF from ESPA rasters")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument("--shapefile", default=str(DEFAULT_SHP), help="Example 6 shapefile")
    args = parser.parse_args()

    extract_all(
        manifest_path=Path(args.manifest),
        shapefile=Path(args.shapefile),
    )


if __name__ == "__main__":
    main()
