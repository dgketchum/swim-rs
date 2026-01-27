#!/usr/bin/env python
"""
Crane (S2) Data Extraction Script
=================================

This script extracts all required input data for the Crane (S2) SWIM-RS example
from Google Earth Engine and GridMET. Run this to reproduce or update the data
used in the tutorial notebooks.

Requirements
------------
- Earth Engine authentication: run `earthengine authenticate` first
- Access to the 'wudr' GCS bucket, OR set --drive to export to Google Drive
- GridMET bias correction TIFs in data/corrections/

Data Extracted
--------------
- ETf: Evapotranspiration fraction from OpenET ensemble (PT-JPL, SIMS, SSEBop, geeSEBAL)
- NDVI: Normalized difference vegetation index from Landsat
- SNODAS SWE: Snow water equivalent
- Irrigation status: From IrrMapper (west) and LANID (east)
- Land cover: From MODIS and FROM-GLC10
- Soils: SSURGO soil properties
- GridMET: Bias-corrected meteorology time series

Usage
-----
    # Extract all data for S2 to GCS bucket:
    python extract_data.py

    # Extract to Google Drive instead:
    python extract_data.py --drive

    # Extract for all flux sites in the shapefile:
    python extract_data.py --all-fields

    # Extract only specific OpenET models:
    python extract_data.py --models ptjpl ssebop

    # Overwrite existing exported data:
    python extract_data.py --overwrite

    # Rebuild local shapefile from master (writes provenance):
    python extract_data.py --build-shp

After Earth Engine tasks complete, sync from bucket:
    gsutil -m rsync -r gs://wudr/3_Crane/ ./

Output Structure
----------------
    data/
    ├── remote_sensing/
    │   └── landsat/extracts/
    │       ├── ptjpl_etf/{irr,inv_irr}/   # PT-JPL ETf CSVs by year
    │       ├── sims_etf/{irr,inv_irr}/    # SIMS ETf CSVs by year
    │       ├── ssebop_etf/{irr,inv_irr}/  # SSEBop ETf CSVs by year
    │       ├── geesebal_etf/{irr,inv_irr}/ # geeSEBAL ETf CSVs by year
    │       └── ndvi/{irr,inv_irr}/         # NDVI CSVs by year
    ├── snow/snodas/extracts/  # SWE CSVs by month
    ├── properties/            # Irrigation, landcover, soils CSVs
    └── met/                   # Meteorology parquet files
"""

import argparse
import os
import sys

import ee

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '../..'))
sys.path.insert(0, ROOT_DIR)

from swimrs.swim.config import ProjectConfig
from swimrs.data_extraction.ee.ptjpl_export import export_ptjpl_zonal_stats
from swimrs.data_extraction.ee.sims_export import export_sims_zonal_stats
from swimrs.data_extraction.ee.ssebop_export import export_ssebop_zonal_stats
from swimrs.data_extraction.ee.geesebal_export import export_geesebal_zonal_stats
from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi
from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe
from swimrs.data_extraction.ee.ee_props import get_irrigation, get_ssurgo, get_landcover
from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.data_extraction.gridmet.gridmet import assign_gridmet_ids, sample_gridmet_corrections, download_gridmet
from swimrs.utils.flux_stations import extract_stations

# OpenET ensemble models available for extraction
OPENET_MODELS = ['ptjpl', 'sims', 'ssebop', 'geesebal']

# Map model names to their export functions
MODEL_EXPORTERS = {
    'ptjpl': export_ptjpl_zonal_stats,
    'sims': export_sims_zonal_stats,
    'ssebop': export_ssebop_zonal_stats,
    'geesebal': export_geesebal_zonal_stats,
}

sys.setrecursionlimit(5000)

# Default: extract only S2 (Crane irrigated alfalfa site)
DEFAULT_FIELDS = ['S2']

# Path to master flux stations shapefile
MASTER_SHAPEFILE = os.path.abspath(os.path.join(ROOT_DIR, 'examples', 'gis', 'flux_stations.shp'))


def init_earth_engine():
    """Initialize Earth Engine, authenticating if needed."""
    print("Initializing Earth Engine...")
    if not is_authorized():
        ee.Authenticate()
    ee.Initialize()
    print("Earth Engine initialized.")


def load_config():
    """Load project configuration from TOML file."""
    config_file = os.path.join(PROJECT_DIR, '3_Crane.toml')
    cfg = ProjectConfig()
    cfg.read_config(config_file, project_root_override=PROJECT_DIR)
    return cfg


def setup_local_shapefile(select_fields, overwrite=False):
    """Extract selected stations from master shapefile to local GIS directory.

    Parameters
    ----------
    select_fields : list of str
        Station IDs to extract (e.g., ['S2'])
    overwrite : bool
        If True, overwrite existing shapefile
    """
    gis_dir = os.path.join(SCRIPT_DIR, 'gis')
    local_shp = os.path.join(gis_dir, 'flux_fields.shp')

    if os.path.exists(local_shp) and not overwrite:
        print(f"Using existing local shapefile: {local_shp}")
        return local_shp

    if not os.path.exists(MASTER_SHAPEFILE):
        raise FileNotFoundError(
            f"Master shapefile not found: {MASTER_SHAPEFILE}\n"
            f"Run: python -m swimrs.utils.flux_stations create --help"
        )

    print(f"\nExtracting stations from master shapefile...")
    extract_stations(
        MASTER_SHAPEFILE,
        select_fields,
        local_shp,
        overwrite=True
    )

    return local_shp


def extract_openet_etf(cfg, select_fields, use_drive, models=None, mask_types=None, overwrite=False):
    """Extract ETf (evapotranspiration fraction) from OpenET ensemble models.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    use_drive : bool
        If True, export to Google Drive; otherwise to GCS bucket
    models : list, optional
        OpenET models to extract. Default: all models (ptjpl, sims, ssebop, geesebal)
    mask_types : list, optional
        Mask types to extract. Default: ['irr', 'inv_irr']
    overwrite : bool
        If True, overwrite existing exported data
    """
    if models is None:
        models = OPENET_MODELS
    if mask_types is None:
        mask_types = ['irr', 'inv_irr']

    # OpenET export functions only support GCS bucket export
    if use_drive:
        print("  Note: OpenET ETf exports only support GCS bucket, ignoring --drive flag")
    bucket = cfg.ee_bucket

    for model in models:
        if model not in MODEL_EXPORTERS:
            print(f"  Warning: Unknown model '{model}', skipping. Available: {list(MODEL_EXPORTERS.keys())}")
            continue

        export_func = MODEL_EXPORTERS[model]
        etf_base = os.path.join(SCRIPT_DIR, 'remote_sensing', 'landsat', 'extracts', f'{model}_etf')

        for mask in mask_types:
            print(f"\nExtracting {model.upper()} ETf ({mask})...")
            check_dir = None if overwrite else os.path.join(etf_base, mask)

            export_func(
                cfg.fields_shapefile,
                bucket=bucket,
                feature_id=cfg.feature_id_col,
                select=select_fields,
                start_yr=cfg.start_dt.year,
                end_yr=cfg.end_dt.year,
                mask_type=mask,
                check_dir=check_dir,
                state_col=cfg.state_col,
                file_prefix=cfg.project_name,
            )


def extract_ndvi(cfg, select_fields, use_drive, satellite='landsat', mask_types=None, overwrite=False):
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
    overwrite : bool
        If True, overwrite existing exported data
    """
    if mask_types is None:
        mask_types = ['irr', 'inv_irr']

    dest = 'drive' if use_drive else 'bucket'
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get('earth_engine', {}).get('drive_folder', cfg.project_name)

    # Check directory for existing exports (use satellite-specific path)
    ndvi_base = os.path.join(SCRIPT_DIR, 'remote_sensing', satellite, 'extracts', 'ndvi')

    for mask in mask_types:
        print(f"\nExtracting {satellite} NDVI ({mask})...")
        mask_check_dir = os.path.join(ndvi_base, mask)
        # Only use check_dir if not overwriting AND directory exists
        check_dir = None if overwrite or not os.path.exists(mask_check_dir) else mask_check_dir
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
            drive_categorize=use_drive
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
    overwrite : bool
        If True, overwrite existing exported data
    """
    print("\nExtracting SNODAS SWE...")
    dest = 'drive' if use_drive else 'bucket'
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get('earth_engine', {}).get('drive_folder', cfg.project_name)

    # Check directory for existing exports
    check_dir = None if overwrite else os.path.join(SCRIPT_DIR, 'snow', 'snodas', 'extracts')

    sample_snodas_swe(
        cfg.fields_shapefile,
        bucket=bucket,
        debug=False,
        check_dir=check_dir,
        overwrite=overwrite,
        feature_id=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive
    )


def extract_properties(cfg, select_fields, use_drive, overwrite=False):
    """Extract irrigation, land cover, and soil properties.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    use_drive : bool
        If True, export to Google Drive; otherwise to GCS bucket
    overwrite : bool
        If True, overwrite existing exported data (currently unused for properties)
    """
    _ = overwrite  # Properties don't have check_dir, always re-export
    dest = 'drive' if use_drive else 'bucket'
    bucket = None if use_drive else cfg.ee_bucket
    drive_folder = cfg.resolved_config.get('earth_engine', {}).get('drive_folder', cfg.project_name)

    print("\nExtracting irrigation data...")
    get_irrigation(
        cfg.fields_shapefile,
        desc=f'{cfg.project_name}_irr',
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        lanid=True,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive
    )

    print("\nExtracting land cover data...")
    get_landcover(
        cfg.fields_shapefile,
        desc=f'{cfg.project_name}_landcover',
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive
    )

    print("\nExtracting SSURGO soil data...")
    get_ssurgo(
        cfg.fields_shapefile,
        desc=f'{cfg.project_name}_ssurgo',
        debug=False,
        selector=cfg.feature_id_col,
        select=select_fields,
        dest=dest,
        bucket=bucket,
        file_prefix=cfg.project_name,
        drive_folder=drive_folder,
        drive_categorize=use_drive
    )


def extract_gridmet(cfg, select_fields, overwrite=False):
    """Assign GridMET cells and download meteorology data.

    Parameters
    ----------
    cfg : ProjectConfig
        Project configuration object
    select_fields : list or None
        Field IDs to extract, or None for all fields
    overwrite : bool
        If True, overwrite existing meteorology data
    """
    print("\nAssigning GridMET cells...")
    assign_gridmet_ids(
        fields=cfg.fields_shapefile,
        fields_join=cfg.gridmet_mapping_shp,
        gridmet_points=cfg.gridmet_centroids,
        field_select=select_fields,
        feature_id=cfg.feature_id_col,
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
        start=cfg.start_dt.strftime('%Y-%m-%d'),
        end=cfg.end_dt.strftime('%Y-%m-%d'),
        target_fields=select_fields,
        overwrite=overwrite,
        feature_id=cfg.feature_id_col
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


def print_summary(use_drive, cfg, sentinel=False):
    """Print extraction summary and next steps."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("""
Expected output directories:
  remote_sensing/landsat/extracts/ptjpl_etf/   - PT-JPL ETf CSVs
  remote_sensing/landsat/extracts/sims_etf/    - SIMS ETf CSVs
  remote_sensing/landsat/extracts/ssebop_etf/  - SSEBop ETf CSVs
  remote_sensing/landsat/extracts/geesebal_etf/ - geeSEBAL ETf CSVs
  remote_sensing/landsat/extracts/ndvi/        - Landsat NDVI CSVs""")
    if sentinel:
        print("  remote_sensing/sentinel/extracts/ndvi/       - Sentinel-2 NDVI CSVs")
    print("""  snow/snodas/extracts/                        - SWE CSVs
  properties/                                  - Irrigation, landcover, soils
  met/                                         - Meteorology parquet files

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
        description='Extract SWIM-RS input data from Earth Engine and GridMET',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--drive', action='store_true',
                        help='Export to Google Drive instead of GCS bucket')
    parser.add_argument('--all-fields', action='store_true',
                        help='Extract all flux sites (default: S2 only)')
    parser.add_argument('--fields', nargs='+', default=None,
                        help='Specific field IDs to extract')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=OPENET_MODELS,
                        help=f'OpenET models to extract (default: all). Choices: {OPENET_MODELS}')
    parser.add_argument('--skip-ee', action='store_true',
                        help='Skip Earth Engine exports (GridMET only)')
    parser.add_argument('--sync-only', action='store_true',
                        help='Only sync from bucket, no new exports')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing exported data')
    parser.add_argument('--build-shp', action='store_true',
                        help='Only rebuild local shapefile from master and exit (writes provenance)')
    parser.add_argument('--sentinel', action='store_true',
                        help='Extract Sentinel-2 NDVI (in addition to or instead of Landsat)')
    parser.add_argument('--sentinel-only', action='store_true',
                        help='Extract only Sentinel-2 NDVI (skip Landsat and ETf)')
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

        # Handle sentinel-only mode
        if args.sentinel_only:
            print("\n" + "=" * 60)
            print("SENTINEL-2 NDVI EXTRACTION ONLY")
            print("=" * 60)
            extract_ndvi(cfg, select_fields, args.drive, satellite='sentinel', overwrite=args.overwrite)

        else:
            etf_models = args.models if args.models else OPENET_MODELS
            print(f"ETf models: {etf_models}")
            # Remote sensing extractions (Earth Engine)
            print("\n" + "=" * 60)
            print("PART A: Remote Sensing Extraction (Earth Engine)")
            print("=" * 60)

            extract_openet_etf(cfg, select_fields, args.drive, models=etf_models, overwrite=args.overwrite)
            extract_ndvi(cfg, select_fields, args.drive, satellite='landsat', overwrite=args.overwrite)

            # Extract Sentinel-2 NDVI if --sentinel flag is set
            if args.sentinel:
                print("\n--- Extracting Sentinel-2 NDVI ---")
                extract_ndvi(cfg, select_fields, args.drive, satellite='sentinel', overwrite=args.overwrite)

            # Properties and snow
            print("\n" + "=" * 60)
            print("PART B: Properties and Snow Extraction (Earth Engine)")
            print("=" * 60)

            extract_snodas(cfg, select_fields, args.drive, overwrite=args.overwrite)
            extract_properties(cfg, select_fields, args.drive, overwrite=args.overwrite)

    # GridMET (local processing) - skip for sentinel-only mode
    if not args.sentinel_only:
        print("\n" + "=" * 60)
        print("PART C: Meteorology Extraction (GridMET)")
        print("=" * 60)

        extract_gridmet(cfg, select_fields, overwrite=args.overwrite)

    # Sync if using bucket (skip for sentinel-only as it goes to different path)
    if not args.drive and not args.sentinel_only:
        print("\n" + "=" * 60)
        print("PART D: Sync from Cloud Storage")
        print("=" * 60)
        sync_from_bucket(cfg, dry_run=True)

    print_summary(args.drive, cfg, sentinel=(args.sentinel or args.sentinel_only))


if __name__ == '__main__':
    main()
