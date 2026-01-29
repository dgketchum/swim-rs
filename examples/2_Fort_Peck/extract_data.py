#!/usr/bin/env python
"""
Fort Peck Data Extraction Script
================================

This script extracts all required input data for the Fort Peck (US-FPe) SWIM-RS
example from Google Earth Engine and GridMET. Run this to reproduce or update
the data used in the tutorial notebooks.

Requirements
------------
- Earth Engine authentication: run `earthengine authenticate` first
- Access to the 'wudr' GCS bucket, OR set USE_DRIVE=True to export to Google Drive
- GridMET bias correction TIFs in data/bias_correction_tif/

Data Extracted
--------------
- ETf (PT-JPL): Evapotranspiration fraction from OpenET PT-JPL algorithm
- NDVI: Normalized difference vegetation index from Landsat
- SNODAS SWE: Snow water equivalent
- Irrigation status: From IrrMapper (west) and LANID (east)
- Land cover: From MODIS and FROM-GLC10
- Soils: SSURGO soil properties
- GridMET: Bias-corrected meteorology time series

Usage
-----
    # Extract all data for US-FPe to GCS bucket:
    python extract_data.py

    # Extract NDVI/SNODAS/properties to Google Drive (ETf always goes to bucket):
    python extract_data.py --drive

    # Extract for all 161 flux sites:
    python extract_data.py --all-fields

    # Rebuild local shapefile from master (writes provenance):
    python extract_data.py --build-shp

After Earth Engine tasks complete, sync from bucket:
    gsutil -m rsync -r gs://wudr/2_Fort_Peck/ ./

Output Structure
----------------
    data/
    ├── remote_sensing/
    │   └── landsat/extracts/
    │       ├── ptjpl_etf/     # ETf CSVs by year (PT-JPL algorithm)
    │       └── ndvi/          # NDVI CSVs by year
    ├── snow/snodas/extracts/  # SWE CSVs by month
    ├── properties/            # Irrigation, landcover, soils CSVs
    └── met_timeseries/gridmet/ # Meteorology parquet files
"""

import argparse
import os
import sys

import ee

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "../.."))
sys.path.insert(0, ROOT_DIR)

from swimrs.data_extraction.ee.ee_props import get_irrigation, get_landcover, get_ssurgo
from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi
from swimrs.data_extraction.ee.ptjpl_export import export_ptjpl_zonal_stats
from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe
from swimrs.data_extraction.gridmet.gridmet import (
    assign_gridmet_ids,
    download_gridmet,
    sample_gridmet_corrections,
)
from swimrs.swim.config import ProjectConfig
from swimrs.utils.flux_stations import extract_stations

sys.setrecursionlimit(5000)

# Default: extract only US-FPe (Fort Peck)
DEFAULT_FIELDS = ["US-FPe"]

# Path to master flux stations shapefile
MASTER_SHAPEFILE = os.path.abspath(os.path.join(ROOT_DIR, "examples", "gis", "flux_stations.shp"))


def init_earth_engine():
    """Initialize Earth Engine, authenticating if needed."""
    print("Initializing Earth Engine...")
    if not is_authorized():
        ee.Authenticate()
    ee.Initialize()
    print("Earth Engine initialized.")


def load_config():
    """Load project configuration from TOML file."""
    config_file = os.path.join(PROJECT_DIR, "2_Fort_Peck.toml")
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_root_override=PROJECT_DIR)
    return cfg


def setup_local_shapefile(select_fields, overwrite=False):
    """Extract selected stations from master shapefile to local GIS directory.

    Parameters
    ----------
    select_fields : list of str
        Station IDs to extract (e.g., ['US-FPe'])
    overwrite : bool
        If True, overwrite existing shapefile
    """
    gis_dir = os.path.join(SCRIPT_DIR, "gis")
    local_shp = os.path.join(gis_dir, "flux_fields.shp")

    if os.path.exists(local_shp) and not overwrite:
        print(f"Using existing local shapefile: {local_shp}")
        return local_shp

    if not os.path.exists(MASTER_SHAPEFILE):
        raise FileNotFoundError(
            f"Master shapefile not found: {MASTER_SHAPEFILE}\n"
            f"Run: python -m swimrs.utils.flux_stations create --help"
        )

    print("\nExtracting stations from master shapefile...")
    extract_stations(MASTER_SHAPEFILE, select_fields, local_shp, overwrite=True)

    return local_shp


def extract_etf(cfg, select_fields, mask_types=None, overwrite=False):
    """Extract ETf (evapotranspiration fraction) from OpenET PT-JPL.

    Uses the new export_ptjpl_zonal_stats function which builds PT-JPL images
    directly from Landsat scenes and exports per-scene zonal means to GCS.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    mask_types : list, optional
        Mask types to extract. Default: ['irr', 'inv_irr']
    overwrite : bool, optional
        If True, re-extract all data ignoring existing files. Default: False
    """
    if mask_types is None:
        mask_types = ["irr", "inv_irr"]

    for mask in mask_types:
        print(f"\nExtracting ETf ({mask}) using PT-JPL...")
        # Check local directory to skip already-extracted years (unless overwrite)
        check_dir = (
            None
            if overwrite
            else os.path.join(
                SCRIPT_DIR, "remote_sensing", "landsat", "extracts", "ptjpl_etf", mask
            )
        )

        export_ptjpl_zonal_stats(
            shapefile=cfg.fields_shapefile,
            bucket=cfg.ee_bucket,
            feature_id=cfg.feature_id_col,
            select=select_fields,
            start_yr=cfg.start_dt.year,
            end_yr=cfg.end_dt.year,
            mask_type=mask,
            check_dir=check_dir,
            state_col=cfg.state_col,
            file_prefix=cfg.project_name,
        )


def extract_ndvi(
    cfg, select_fields, use_drive, satellite="landsat", mask_types=None, overwrite=False
):
    """Extract NDVI from Landsat or Sentinel-2.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    use_drive : bool
        If True, export to Google Drive; otherwise to GCS bucket
    satellite : str
        'landsat' or 'sentinel'
    mask_types : list, optional
        Mask types to extract. Default: ['irr', 'inv_irr']
    overwrite : bool, optional
        If True, re-extract all data ignoring existing files. Default: False
    """
    if mask_types is None:
        mask_types = ["irr", "inv_irr"]

    dest = "drive" if use_drive else "bucket"
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get("earth_engine", {}).get("drive_folder", cfg.project_name)

    for mask in mask_types:
        print(f"\nExtracting {satellite} NDVI ({mask})...")
        # Check local directory to skip already-extracted years (unless overwrite)
        check_dir = (
            None
            if overwrite
            else os.path.join(SCRIPT_DIR, "remote_sensing", satellite, "extracts", "ndvi", mask)
        )
        sparse_sample_ndvi(
            cfg.fields_shapefile,
            bucket=bucket,
            debug=False,
            mask_type=mask,
            check_dir=check_dir,
            start_yr=cfg.start_dt.year,
            end_yr=cfg.end_dt.year,
            feature_id=cfg.feature_id_col,
            select=select_fields,
            state_col=cfg.state_col,
            satellite=satellite,
            dest=dest,
            file_prefix=cfg.project_name,
            drive_folder=drive_folder,
            drive_categorize=use_drive,
        )


def extract_snodas(cfg, select_fields, use_drive, overwrite=False):
    """Extract SNODAS snow water equivalent.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    use_drive : bool
        If True, export to Google Drive; otherwise to GCS bucket
    overwrite : bool, optional
        If True, re-extract all data ignoring existing files. Default: False
    """
    print("\nExtracting SNODAS SWE...")
    dest = "drive" if use_drive else "bucket"
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get("earth_engine", {}).get("drive_folder", cfg.project_name)

    # Check local directory to skip already-extracted months (unless overwrite)
    check_dir = None if overwrite else os.path.join(SCRIPT_DIR, "snow", "snodas", "extracts")
    sample_snodas_swe(
        cfg.fields_shapefile,
        bucket=bucket,
        debug=False,
        check_dir=check_dir,
        overwrite=False,
        feature_id=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive,
    )


def extract_properties(cfg, select_fields, use_drive):
    """Extract irrigation, land cover, and soil properties."""
    dest = "drive" if use_drive else "bucket"
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get("earth_engine", {}).get("drive_folder", cfg.project_name)

    print("\nExtracting irrigation data...")
    get_irrigation(
        cfg.fields_shapefile,
        desc=f"{cfg.project_name}_irr",
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        lanid=True,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive,
    )

    print("\nExtracting land cover data...")
    get_landcover(
        cfg.fields_shapefile,
        desc=f"{cfg.project_name}_landcover",
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive,
    )

    print("\nExtracting SSURGO soil data...")
    get_ssurgo(
        cfg.fields_shapefile,
        desc=f"{cfg.project_name}_ssurgo",
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive,
    )


def extract_gridmet(cfg, select_fields, overwrite=False):
    """Assign GridMET cells and download meteorology data."""
    print("\nAssigning GridMET cells...")
    assign_gridmet_ids(
        fields=cfg.fields_shapefile,
        fields_join=cfg.gridmet_mapping_shp,
        gridmet_points=cfg.gridmet_centroids,
        feature_id=cfg.feature_id_col,
        field_select=select_fields,
    )

    print("\nExtracting bias correction factors...")
    sample_gridmet_corrections(
        fields_join=cfg.gridmet_mapping_shp,
        gridmet_ras=cfg.correction_tifs,
        factors_js=cfg.gridmet_factors,
    )

    print("\nDownloading GridMET data...")
    os.makedirs(cfg.met_dir, exist_ok=True)
    download_gridmet(
        cfg.gridmet_mapping_shp,
        cfg.gridmet_factors,
        cfg.met_dir,
        start=cfg.start_dt.strftime("%Y-%m-%d"),
        end=cfg.end_dt.strftime("%Y-%m-%d"),
        target_fields=select_fields,
        overwrite=overwrite,
        feature_id=cfg.feature_id_col,
    )


def sync_from_bucket(cfg, dry_run=False):
    """Sync Earth Engine exports from GCS bucket to local filesystem."""
    if cfg.ee_bucket:
        if dry_run:
            print("\nPreview of files to sync (dry run):")
        else:
            print("\nSyncing data from bucket...")
        cfg.sync_from_bucket(dry_run=dry_run)
        if not dry_run:
            print("Sync complete!")


def print_summary(use_drive, cfg):
    """Print extraction summary and next steps."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("""
Expected output directories:
  remote_sensing/landsat/extracts/ptjpl_etf/  - ETf CSVs (PT-JPL)
  remote_sensing/landsat/extracts/ndvi/       - NDVI CSVs
  snow/snodas/extracts/                       - SWE CSVs
  properties/                                 - Irrigation, landcover, soils
  met_timeseries/gridmet/                     - Meteorology parquet files

Next steps:
1. Monitor Earth Engine tasks: https://code.earthengine.google.com/tasks
2. Once complete, sync data:""")

    if use_drive:
        print("   Download from Google Drive and place in data/ subdirectories")
    else:
        print(f"   gsutil -m rsync -r gs://{cfg.ee_bucket}/{cfg.project_name}/ {cfg.data_dir}/")

    print("3. Run the tutorial notebooks to process the data")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SWIM-RS input data from Earth Engine and GridMET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Export NDVI/SNODAS/properties to Google Drive (ETf always uses bucket)",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Extract all 161 flux sites (default: US-FPe only)",
    )
    parser.add_argument("--fields", nargs="+", default=None, help="Specific field IDs to extract")
    parser.add_argument(
        "--skip-ee", action="store_true", help="Skip Earth Engine exports (GridMET only)"
    )
    parser.add_argument(
        "--sync-only", action="store_true", help="Only sync from bucket, no new exports"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-extract all data, ignoring existing files (default: False)",
    )
    parser.add_argument(
        "--build-shp",
        action="store_true",
        help="Only rebuild local shapefile from master and exit (writes provenance)",
    )
    args = parser.parse_args()

    # Determine which fields to extract
    if args.fields:
        select_fields = args.fields
    elif args.all_fields:
        select_fields = None
    else:
        select_fields = DEFAULT_FIELDS

    # Setup local shapefile from master (if extracting specific fields)
    if select_fields is not None:
        if args.build_shp:
            setup_local_shapefile(select_fields, overwrite=True)
            return
        else:
            setup_local_shapefile(select_fields, overwrite=False)

    # Load config (needed for all paths)
    cfg = load_config()

    # Print configuration
    print(f"\nProject: {cfg.project_name}")
    print(f"Date range: {cfg.start_dt.date()} to {cfg.end_dt.date()}")
    print(f"Fields: {select_fields if select_fields else 'ALL'}")

    if args.sync_only:
        init_earth_engine()
        print(f"Bucket: {cfg.ee_bucket}")
        sync_from_bucket(cfg, dry_run=False)
        return

    if not args.skip_ee:
        # Initialize Earth Engine only when needed
        init_earth_engine()
        print(f"Bucket: {cfg.ee_bucket}")
        print(f"Destination: {'Google Drive' if args.drive else 'GCS bucket'}")
        # Remote sensing extractions (Earth Engine)
        print("\n" + "=" * 60)
        print("PART A: Remote Sensing Extraction (Earth Engine)")
        print("=" * 60)

        extract_etf(cfg, select_fields, overwrite=args.overwrite)
        extract_ndvi(cfg, select_fields, args.drive, overwrite=args.overwrite)

        # Properties and snow
        print("\n" + "=" * 60)
        print("PART B: Properties and Snow Extraction (Earth Engine)")
        print("=" * 60)

        extract_snodas(cfg, select_fields, args.drive, overwrite=args.overwrite)
        extract_properties(cfg, select_fields, args.drive)

    # GridMET (local processing)
    print("\n" + "=" * 60)
    print("PART C: Meteorology Extraction (GridMET)")
    print("=" * 60)

    extract_gridmet(cfg, select_fields, overwrite=args.overwrite)

    # Sync if using bucket
    if not args.drive:
        print("\n" + "=" * 60)
        print("PART D: Sync from Cloud Storage")
        print("=" * 60)
        sync_from_bucket(cfg, dry_run=True)

    print_summary(args.drive, cfg)


if __name__ == "__main__":
    main()
